"""
Gateway Manager - Manages connections to multiple Powerwall gateways.

This is the central hub of the server that manages all pypowerwall connections,
performs background polling, caches data, and provides fast API responses.

Architecture:
    - Singleton pattern (single gateway_manager instance)
    - Background polling task runs every PW_CACHE_EXPIRE seconds (default: 5s)
    - Concurrent polling of all gateways using asyncio.gather
    - Cached data for instant API responses without blocking
    - Automatic reconnection on failure

Connection Modes:
    TEDAPI (Local Gateway):
        pw = pypowerwall.Powerwall(
            host="192.168.91.1",
            gw_pwd="gateway_wifi_password",
            timeout=3
        )
    
    Cloud Mode:
        pw = pypowerwall.Powerwall(
            email="user@example.com",
            authpath="/path/to/auth/files",
            cloudmode=True
        )
    
    FleetAPI:
        pw = pypowerwall.Powerwall(
            email="user@example.com",
            authpath="/path/to/auth/files",
            fleetapi=True
        )

Data Flow:
    1. Background task calls _poll_gateway() for each gateway every N seconds
    2. _poll_gateway() makes blocking pypowerwall calls in executor with timeouts
    3. Results cached in self.cache[gateway_id] as GatewayStatus objects
    4. API endpoints read from cache (instant response, no blocking)
    5. Failed polls update gateway status to offline (automatic retry next cycle)

Error Handling:
    - Connection failures logged but don't crash server
    - Timeouts on pypowerwall calls (3-10s depending on operation)
    - Offline gateways excluded from aggregates
    - Cached data remains available during outages
    - Automatic reconnection every poll cycle

Thread Safety:
    - All operations use asyncio (no threads/locks needed for reads)
    - Write operations serialized via _write_lock to prevent set_operation() races
    - Single event loop handles all concurrency
    - Background task coordinated via asyncio.create_task()
    - Graceful shutdown via task cancellation

Performance:
    - Concurrent gateway polling for speed
    - Short timeouts prevent blocking
    - Cached responses for instant API access
    - Minimal memory footprint (only latest data cached)
"""
import asyncio
import json
import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import pypowerwall
from app.models.gateway import Gateway, GatewayStatus, PowerwallData, AggregateData
from app.core.scaling import raw_to_tesla_battery_percent
from app.config import GatewayConfig

logger = logging.getLogger(__name__)

# Methods that write gateway state and must not run concurrently.
# set_operation() always writes backup_reserve_percent + real_mode together,
# reading the field it wasn't given from the 5-second poll cache. Two
# concurrent writes both see the same stale cache value and the last one to
# land on the gateway clobbers whichever field it didn't own.
_WRITE_METHODS = frozenset(
    {
        "set_reserve",
        "set_mode",
        "set_operation",
        "set_grid_charging",
        "set_grid_export",
    }
)

class GatewayManager:
    """Manages multiple Powerwall gateway connections."""

    def __init__(self):
        self.gateways: Dict[str, Gateway] = {}
        self.connections: Dict[str, pypowerwall.Powerwall] = {}
        self.cache: Dict[str, GatewayStatus] = {}
        self._poll_task: Optional[asyncio.Task] = None
        self._poll_interval = 5  # Default, will be set from config during initialize()

        # Exponential backoff tracking per gateway
        self._consecutive_failures: Dict[str, int] = {}  # Track failure count
        self._next_poll_time: Dict[
            str, float
        ] = {}  # Track when to poll next (Unix timestamp)
        self._last_successful_data: Dict[
            str, PowerwallData
        ] = {}  # Keep last good data for graceful degradation
        self._pending_configs: Dict[
            str, GatewayConfig
        ] = {}  # Gateways waiting for lazy initialization
        self._preserve_stale_count: Dict[str, int] = {}  # Multi-PW snapshot preservation staleness tracker

        # Dedicated thread pool for blocking pypowerwall operations
        # Will be sized during initialize() based on gateway count
        self._executor: Optional[ThreadPoolExecutor] = None

        # Serializes concurrent write operations to prevent set_operation()
        # from reading stale cache when two control calls race.
        self._write_lock: asyncio.Lock = asyncio.Lock()

        # Cloud connection for control operations (set_reserve, set_mode).
        # TEDAPI doesn't support POST/write APIs, so a separate cloud-mode
        # pypowerwall instance is created when cloud credentials are available
        # alongside a TEDAPI gateway. This enables hybrid operation:
        # TEDAPI for fast local reads, cloud for control writes.
        self._cloud_control: Optional[pypowerwall.Powerwall] = None

    @staticmethod
    def _expected_battery_block_count(data: Optional[PowerwallData]) -> int:
        """Return expected battery block count from cached TEDAPI config when available."""
        if not data or not isinstance(data.tedapi_config, dict):
            return 0
        battery_blocks = data.tedapi_config.get("battery_blocks") or []
        return len(battery_blocks) if isinstance(battery_blocks, list) else 0

    @staticmethod
    def _count_tepinv_devices(vitals: Optional[Dict[str, Any]]) -> int:
        """Count Powerwall inverter entries in vitals payload."""
        if not isinstance(vitals, dict):
            return 0
        return sum(1 for key in vitals if key.startswith("TEPINV--"))

    @staticmethod
    def _count_system_status_blocks(system_status: Optional[Dict[str, Any]]) -> int:
        """Count battery blocks in cached system status payload."""
        if not isinstance(system_status, dict):
            return 0
        battery_blocks = system_status.get("battery_blocks") or []
        return len(battery_blocks) if isinstance(battery_blocks, list) else 0

    # Max consecutive polls a preserved snapshot is promoted before letting
    # partial data through.  Prevents stale follower data from living forever.
    _PRESERVE_STALENESS_CAP = 3

    def _preserve_complete_multi_pw_snapshot(
        self, gateway_id: str, data: PowerwallData
    ) -> PowerwallData:
        """Avoid downgrading a complete multi-PW TEDAPI snapshot with a partial one.

        TEDAPI multi-PW data is synthesized in pypowerwall from follower vitals.
        If a single poll cycle drops follower data, the server can otherwise cache a
        one-PW snapshot even though the gateway config still reports multiple blocks.

        The guard is capped at ``_PRESERVE_STALENESS_CAP`` consecutive polls — after
        that, the partial data passes through so downstream consumers see reality.
        """
        previous = self._last_successful_data.get(gateway_id)
        if not previous:
            return data

        current_config_count = self._expected_battery_block_count(data)
        previous_config_count = self._expected_battery_block_count(previous)

        # Trust the current TEDAPI config when it is present and non-zero.
        # Only fall back to the previous (larger) count when the current config
        # is missing — that indicates a transient read failure, not a legitimate
        # transition from multi-PW to single-PW.
        if current_config_count > 0:
            expected_blocks = current_config_count
        else:
            expected_blocks = previous_config_count

        if expected_blocks < 2:
            # Not a multi-PW system (or no longer one) — reset staleness counter.
            self._preserve_stale_count.pop(gateway_id, None)
            return data

        # --- Staleness cap ---
        stale_key = gateway_id
        stale_count = self._preserve_stale_count.get(stale_key, 0)
        if stale_count >= self._PRESERVE_STALENESS_CAP:
            logger.warning(
                "Preservation guard for %s hit staleness cap (%d polls) — "
                "letting partial data through",
                gateway_id,
                self._PRESERVE_STALENESS_CAP,
            )
            self._preserve_stale_count.pop(stale_key, None)
            return data

        preserved_any = False

        current_vitals_count = self._count_tepinv_devices(data.vitals)
        previous_vitals_count = self._count_tepinv_devices(previous.vitals)
        current_status_count = self._count_system_status_blocks(data.system_status)
        previous_status_count = self._count_system_status_blocks(previous.system_status)

        if (
            current_vitals_count < expected_blocks
            and previous_vitals_count >= expected_blocks
        ):
            logger.warning(
                "Preserving prior complete vitals snapshot for %s: "
                "expected %d TEPINV blocks, got %d in current poll (stale %d/%d)",
                gateway_id,
                expected_blocks,
                current_vitals_count,
                stale_count + 1,
                self._PRESERVE_STALENESS_CAP,
            )
            data.vitals = deepcopy(previous.vitals)
            preserved_any = True

        if (
            current_status_count < expected_blocks
            and previous_status_count >= expected_blocks
        ):
            logger.warning(
                "Preserving prior complete system_status snapshot for %s: "
                "expected %d battery blocks, got %d in current poll (stale %d/%d)",
                gateway_id,
                expected_blocks,
                current_status_count,
                stale_count + 1,
                self._PRESERVE_STALENESS_CAP,
            )
            data.system_status = deepcopy(previous.system_status)
            preserved_any = True

        if not data.tedapi_config and previous.tedapi_config:
            data.tedapi_config = deepcopy(previous.tedapi_config)

        if preserved_any:
            self._preserve_stale_count[stale_key] = stale_count + 1
        else:
            # Current poll was complete — reset staleness.
            self._preserve_stale_count.pop(stale_key, None)

        return data

    async def initialize(
        self, gateway_configs: List[GatewayConfig], poll_interval: int = 5
    ):
        """Initialize gateway manager - non-blocking.

        This method sets up gateways for lazy initialization. Actual pypowerwall
        connections are created during the first poll cycle to ensure the server
        starts accepting connections immediately.

        Args:
            gateway_configs: List of gateway configurations
            poll_interval: Polling frequency in seconds (from PW_CACHE_EXPIRE, default: 5)
        """
        self._poll_interval = poll_interval

        # Size thread pool based on gateway count
        # Formula: max(10, num_gateways * 3) to support concurrent API calls
        num_gateways = len(gateway_configs)
        pool_size = max(10, num_gateways * 3)
        self._executor = ThreadPoolExecutor(
            max_workers=pool_size, thread_name_prefix="pypowerwall"
        )
        logger.info(
            f"Thread pool initialized with {pool_size} workers for {num_gateways} gateway(s)"
        )

        for config in gateway_configs:
            try:
                # Validate configuration
                # TEDAPI mode: need host + (gw_pwd OR rsa_key_path)
                # Cloud mode: need email (authpath is optional, pypowerwall has defaults)
                has_tedapi = config.host and (config.gw_pwd or config.rsa_key_path)
                has_cloud = config.email  # cloud_mode is auto-set, email is sufficient

                if not (has_tedapi or has_cloud):
                    logger.error(
                        f"Invalid configuration for gateway {config.id}: need host+gw_pwd or host+rsa_key_path (TEDAPI) or email (Cloud)"
                    )
                    continue

                # Warn when both gw_pwd and rsa_key_path are set alongside host.
                # pypowerwall selects TEDAPI v1r (RSA auth) in this case, which
                # limits follower Powerwall data to primary-only unless a wifi_host
                # is also provided for the follower WiFi fallback path.
                if config.host and config.gw_pwd and config.rsa_key_path and not config.wifi_host:
                    logger.warning(
                        "Gateway %s: PW_HOST + PW_GW_PWD + PW_RSA_KEY_PATH are "
                        "all set — TEDAPI v1r mode is active but follower "
                        "Powerwall data will be limited to the primary unit only. "
                        "To see all Powerwalls, either: "
                        "(a) set PW_WIFI_HOST=<gateway-ip> to enable WiFi fallback "
                        "for follower queries while keeping v1r, or "
                        "(b) remove PW_RSA_KEY_PATH to use TEDAPI WiFi mode.",
                        config.id,
                    )

                # Auto-enable cloud_mode if email is set but no host
                if config.email and not config.host:
                    config.cloud_mode = True

                gateway = Gateway(
                    id=config.id,
                    name=config.name,
                    host=config.host,
                    port=config.port,
                    gw_pwd=config.gw_pwd,
                    rsa_key_path=config.rsa_key_path,
                    rsa_key_configured=bool(config.rsa_key_path),
                    wifi_host=config.wifi_host,
                    email=config.email,
                    timezone=config.timezone,
                    cloud_mode=config.cloud_mode,
                    fleetapi=config.fleetapi,
                    type=config.type,
                )

                # Store gateway - connection will be created lazily on first poll
                self.gateways[config.id] = gateway
                self._pending_configs[config.id] = config  # All start as pending

                self.cache[config.id] = GatewayStatus(
                    gateway=gateway, online=False, error="Initializing..."
                )

                # Initialize backoff tracking
                self._consecutive_failures[config.id] = 0
                self._next_poll_time[config.id] = 0  # Poll immediately

                # Determine and log connection mode
                if config.fleetapi:
                    mode = "FleetAPI"
                elif config.cloud_mode:
                    mode = "Cloud"
                else:
                    mode = "TEDAPI"

                logger.info(
                    f"Registered gateway: {config.id} ({config.name}) - {mode} mode - connection pending"
                )
            except Exception as e:
                logger.error(f"Failed to initialize gateway {config.id}: {e}")

        # Initialize cloud control connection for TEDAPI gateways with cloud credentials.
        # This enables hybrid operation: local TEDAPI reads + cloud control writes.
        from app.config import settings
        for config in gateway_configs:
            if config.host and config.gw_pwd and config.email and not config.cloud_mode:
                try:
                    authpath = config.authpath or settings.pw_authpath or ""
                    loop = asyncio.get_running_loop()
                    cloud_kwargs = {
                        "email": config.email,
                        "authpath": authpath,
                        "cachefile": "/tmp/.powerwall.cloud",
                        "timezone": config.timezone,
                        "fleetapi": config.fleetapi,
                        "auto_select": True,
                    }
                    self._cloud_control = await asyncio.wait_for(
                        loop.run_in_executor(
                            self._executor,
                            lambda kw=cloud_kwargs: pypowerwall.Powerwall(**kw),
                        ),
                        timeout=15.0,
                    )
                    logger.info(
                        "Cloud control connection established for write operations"
                    )
                    break  # Only need one cloud control connection
                except Exception as e:
                    logger.warning(
                        f"Cloud control connection failed (control will be unavailable): {e}"
                    )

        # Start polling task
        if self.gateways:
            self._poll_task = asyncio.create_task(self._poll_gateways())
            logger.info(
                f"Gateway manager ready - {len(self.gateways)} gateway(s) will connect on first poll"
            )

    async def shutdown(self):
        """Shutdown gateway manager and cleanup resources."""
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                # Expected when cancelling the polling task during shutdown
                pass

        # Shutdown thread pool executor
        if self._executor:
            self._executor.shutdown(wait=False)
        logger.info("Gateway manager shutdown complete")

    async def _poll_gateways(self):
        """Background task to poll all gateways on a fixed cadence.

        Uses a fixed-tick scheduling pattern to ensure consistent poll-to-poll
        intervals regardless of how long the actual poll takes. With the previous
        sleep-after-poll approach the effective interval was poll_duration + sleep,
        causing drift (e.g. ~8-9s actual interval when PW_CACHE_EXPIRE=5).

        The loop records the monotonic clock time (via ``loop.time()``) at the
        start of each cycle and only sleeps for the *remaining* time after all
        gateways have been polled, so the next cycle starts as close to the
        configured interval as possible.
        """
        while True:
            try:
                loop = asyncio.get_running_loop()
                loop_start = loop.time()

                # Poll all gateways concurrently
                tasks = [
                    self._poll_gateway(gateway_id)
                    for gateway_id in self.gateways.keys()
                ]
                await asyncio.gather(*tasks, return_exceptions=True)

                # Sleep only the remaining time to maintain fixed interval
                elapsed = loop.time() - loop_start
                sleep_time = max(0, self._poll_interval - elapsed)
                await asyncio.sleep(sleep_time)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in polling task: {e}")
                await asyncio.sleep(self._poll_interval)

    async def _poll_gateway(self, gateway_id: str) -> None:
        """Poll a single gateway for data with exponential backoff on failures."""
        try:
            # Check if we're in backoff period
            now = datetime.now().timestamp()
            next_poll = self._next_poll_time.get(gateway_id, 0)

            if now < next_poll:
                # Skip this poll cycle - in backoff period
                logger.debug(
                    f"Gateway {gateway_id} in backoff, skipping poll (next poll at {next_poll - now:.0f}s)"
                )
                return

            # Check for lazy initialization - create connection if pending
            if (
                gateway_id in self._pending_configs
                and gateway_id not in self.connections
            ):
                config = self._pending_configs[gateway_id]
                logger.info(f"Attempting lazy initialization of gateway {gateway_id}")

                from app.config import settings

                loop = asyncio.get_running_loop()

                try:
                    if config.cloud_mode and config.email:
                        cloud_kwargs = {
                            "email": config.email,
                            "cloudmode": True,
                            "fleetapi": config.fleetapi,
                            "timezone": config.timezone,
                        }
                        if config.authpath:
                            cloud_kwargs["authpath"] = config.authpath
                        pw = await asyncio.wait_for(
                            loop.run_in_executor(
                                self._executor,
                                lambda kw=cloud_kwargs: pypowerwall.Powerwall(**kw),
                            ),
                            timeout=15.0,
                        )
                        connected = await asyncio.wait_for(
                            loop.run_in_executor(self._executor, pw.is_connected),
                            timeout=10.0,
                        )
                        if not connected:
                            raise Exception(
                                f"pypowerwall failed to connect to gateway {gateway_id} (cloud mode)"
                            )
                    else:
                        # Build host string with optional non-standard port
                        # e.g. host="192.168.1.50", port=8443 -> "192.168.1.50:8443"
                        effective_host = (
                            f"{config.host}:{config.port}" if config.port else config.host
                        )
                        tedapi_kwargs = {
                            "host": effective_host,
                            "gw_pwd": config.gw_pwd,
                            "timezone": config.timezone,
                            "timeout": settings.timeout,
                            "poolmaxsize": settings.pool_maxsize,
                        }
                        if settings.pw_password:
                            tedapi_kwargs["password"] = settings.pw_password
                        if config.email:
                            tedapi_kwargs["email"] = config.email
                        if config.authpath:
                            tedapi_kwargs["authpath"] = config.authpath
                        if settings.cache_file:
                            tedapi_kwargs["cachefile"] = settings.cache_file
                        if settings.siteid:
                            tedapi_kwargs["siteid"] = settings.siteid
                        if config.rsa_key_path:
                            tedapi_kwargs["rsa_key_path"] = config.rsa_key_path
                        if config.wifi_host:
                            tedapi_kwargs["wifi_host"] = config.wifi_host
                        pw = await asyncio.wait_for(
                            loop.run_in_executor(
                                self._executor,
                                lambda kw=tedapi_kwargs: pypowerwall.Powerwall(**kw),
                            ),
                            timeout=15.0,
                        )
                        connected = await asyncio.wait_for(
                            loop.run_in_executor(self._executor, pw.is_connected),
                            timeout=10.0,
                        )
                        if not connected:
                            raise Exception(
                                f"pypowerwall failed to connect to gateway {gateway_id} (TEDAPI mode)"
                            )

                    self.connections[gateway_id] = pw
                    del self._pending_configs[gateway_id]

                    # Try to get site_id for cloud mode gateways
                    gateway = self.gateways[gateway_id]
                    if gateway.fleetapi:
                        mode_label = "FleetAPI"
                    elif gateway.cloud_mode:
                        mode_label = "Cloud"
                    elif config.rsa_key_path and config.wifi_host:
                        mode_label = "TEDAPI v1r + WiFi"
                    elif config.rsa_key_path:
                        mode_label = "TEDAPI v1r"
                    else:
                        mode_label = "TEDAPI WiFi"

                    if gateway.cloud_mode or gateway.fleetapi:
                        try:
                            site_id = getattr(pw, "siteid", None) or getattr(
                                pw, "site_id", None
                            )
                            if site_id:
                                gateway.site_id = str(site_id)
                                logger.info(
                                    f"Connected to gateway {gateway_id} - {mode_label} mode (Site ID: {site_id}, Email: {gateway.email})"
                                )
                            else:
                                logger.info(
                                    f"Connected to gateway {gateway_id} - {mode_label} mode (Email: {gateway.email})"
                                )
                        except Exception:
                            logger.info(
                                f"Connected to gateway {gateway_id} - {mode_label} mode"
                            )
                    else:
                        logger.info(
                            f"Connected to gateway {gateway_id} - {mode_label} mode ({gateway.host})"
                        )

                except asyncio.TimeoutError:
                    logger.warning(
                        f"Lazy initialization timeout for gateway {gateway_id} - will retry next cycle"
                    )
                    raise Exception("Connection initialization timeout")
                except Exception as e:
                    logger.warning(
                        f"Lazy initialization failed for gateway {gateway_id}: {e}"
                    )
                    raise

            pw = self.connections.get(gateway_id)
            if not pw:
                logger.debug(
                    f"No connection object for gateway {gateway_id} - waiting for lazy init"
                )
                raise Exception("Connection not yet initialized")

            # Run blocking pypowerwall calls in dedicated executor with timeout protection
            loop = asyncio.get_running_loop()

            # Fetch core data - aggregates is required, vitals/strings are optional
            # Use asyncio.wait_for to timeout if pypowerwall hangs
            try:
                aggregates = await asyncio.wait_for(
                    loop.run_in_executor(
                        self._executor, pw.poll, "/api/meters/aggregates"
                    ),
                    timeout=10.0,  # 10 second timeout
                )
            except asyncio.TimeoutError:
                raise Exception(f"Timeout fetching aggregates from {gateway_id}")
            except Exception as e:
                # If we can't get aggregates, this is a real connection failure
                raise Exception(f"Failed to fetch aggregates: {e}")

            # Apply negative solar correction if configured (PW_NEG_SOLAR=no)
            # This is done at fetch time so all endpoints get consistent data
            from app.config import settings

            if aggregates and not settings.neg_solar:
                solar_power = aggregates.get("solar", {}).get("instant_power", 0)
                if solar_power < 0:
                    # Shift negative solar energy to load
                    if "load" in aggregates and "instant_power" in aggregates["load"]:
                        aggregates["load"]["instant_power"] -= solar_power
                    # Clamp solar to 0
                    if "solar" in aggregates:
                        aggregates["solar"]["instant_power"] = 0
                    logger.debug(
                        f"Applied neg_solar correction for {gateway_id}: solar clamped to 0"
                    )

            # Build PowerwallData with required aggregates
            data = PowerwallData(
                aggregates=aggregates, timestamp=datetime.now().timestamp()
            )

            # Try to get optional vitals and strings (don't fail if these aren't available)
            try:
                data.vitals = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, pw.vitals), timeout=10.0
                )
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug(f"Vitals not available for {gateway_id}: {e}")

            try:
                data.strings = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, pw.strings), timeout=10.0
                )
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug(f"Strings not available for {gateway_id}: {e}")

            # Try to get additional data
            try:
                data.soe_raw = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, pw.level), timeout=5.0
                )
                data.soe = raw_to_tesla_battery_percent(data.soe_raw)
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug(f"SOE not available for {gateway_id}: {e}")

            try:
                data.freq = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, pw.freq), timeout=5.0
                )
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug(f"Frequency not available for {gateway_id}: {e}")

            try:
                data.status = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, pw.status), timeout=5.0
                )
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug(f"Status not available for {gateway_id}: {e}")

            try:
                data.version = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, pw.version), timeout=5.0
                )
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug(f"Version not available for {gateway_id}: {e}")

            try:
                data.din = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, pw.din), timeout=5.0
                )
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug(f"DIN not available for {gateway_id}: {e}")

            try:
                data.uptime = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, pw.uptime), timeout=5.0
                )
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug(f"Uptime not available for {gateway_id}: {e}")

            logger.debug(f"Gateway {gateway_id} aggregates: {data.aggregates}")

            # Try to get alerts (for caching)
            try:
                data.alerts = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, pw.alerts), timeout=5.0
                )
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug(f"Alerts not available for {gateway_id}: {e}")

            # Try to get temps (for caching)
            try:
                data.temps = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, pw.temps), timeout=5.0
                )
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug(f"Temps not available for {gateway_id}: {e}")

            # Try to get site name (for caching)
            try:
                data.site_name = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, pw.site_name), timeout=5.0
                )
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug(f"Site name not available for {gateway_id}: {e}")

            # Detect PW3 status from pypowerwall TEDAPI connection
            try:
                if hasattr(pw, "tedapi") and pw.tedapi:
                    pw3_status = getattr(pw.tedapi, "pw3", None)
                    if pw3_status is not None:
                        data.pw3 = bool(pw3_status)
                    # Also cache tedapi_mode
                    if hasattr(pw, "tedapi_mode"):
                        data.tedapi_mode = pw.tedapi_mode
            except Exception:
                pass

            # Cache TEDAPI config for battery block type enrichment (PW3 systems)
            # battery_blocks[].type gives "Powerwall3" / "Powerwall3Follower" etc.,
            # which is more useful for model detection than system_status Type ("ACPW").
            try:
                if hasattr(pw, "tedapi") and pw.tedapi and hasattr(pw.tedapi, "get_config"):
                    tedapi_config = await asyncio.wait_for(
                        loop.run_in_executor(self._executor, pw.tedapi.get_config),
                        timeout=10.0,
                    )
                    if tedapi_config and isinstance(tedapi_config, dict):
                        data.tedapi_config = tedapi_config
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug(f"TEDAPI config not available for {gateway_id}: {e}")

            # Try to get grid status (for caching)
            try:
                data.grid_status = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, pw.grid_status), timeout=5.0
                )
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug(f"Grid status not available for {gateway_id}: {e}")

            # Try to get detailed grid status from API (for /api/system_status/grid_status endpoint)
            try:
                grid_status_response = await asyncio.wait_for(
                    loop.run_in_executor(
                        self._executor, pw.poll, "/api/system_status/grid_status"
                    ),
                    timeout=5.0,
                )
                if isinstance(grid_status_response, str):
                    data.grid_status_detail = json.loads(grid_status_response)
                else:
                    data.grid_status_detail = grid_status_response
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug(f"Grid status detail not available for {gateway_id}: {e}")

            # Try to get operation mode (for /api/operation endpoint)
            # Mode can be "self_consumption", "backup", or "autonomous" (time-based control).
            # This is fetched from /api/operation on the gateway via pw.get_mode() and
            # must be polled on every cycle so that mode changes made in the Tesla app
            # are reflected promptly (fixes issue #14).
            # Pre-fill from last known good value so a transient failure doesn't wipe the cache.
            last_data = self._last_successful_data.get(gateway_id)
            if last_data and last_data.mode:
                data.mode = last_data.mode
            try:
                data.mode = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, pw.get_mode),
                    timeout=5.0,
                )
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug(f"Operation mode not available for {gateway_id}: {e}")

            # Try to get reserve and time remaining (for caching)
            try:
                # Request the Tesla App scaled reserve setting (scale=True)
                data.reserve = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, lambda: pw.get_reserve(scale=True)), timeout=5.0
                )
                data.time_remaining = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, pw.get_time_remaining),
                    timeout=5.0,
                )
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug(
                    f"Reserve/time remaining not available for {gateway_id}: {e}"
                )

            # Try to get system status for /pod endpoint (for caching)
            try:
                data.system_status = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, pw.system_status), timeout=5.0
                )
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug(f"System status not available for {gateway_id}: {e}")

            # Try to get fan speeds for /fans endpoint (TEDAPI only)
            try:
                if hasattr(pw, "get_fan_speeds"):
                    data.fan_speeds = await asyncio.wait_for(
                        loop.run_in_executor(self._executor, pw.get_fan_speeds),
                        timeout=5.0,
                    )
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug(f"Fan speeds not available for {gateway_id}: {e}")

            # Try to get networks for /api/system/networks endpoint
            try:
                networks_result = await asyncio.wait_for(
                    loop.run_in_executor(
                        self._executor, lambda: pw.poll("/api/networks")
                    ),
                    timeout=5.0,
                )
                if networks_result and isinstance(networks_result, list):
                    data.networks = networks_result
                elif networks_result and isinstance(networks_result, str):
                    try:
                        data.networks = json.loads(networks_result)
                    except json.JSONDecodeError:
                        pass
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug(f"Networks not available for {gateway_id}: {e}")

            # Try to get powerwalls for /api/powerwalls endpoint
            try:
                powerwalls_result = await asyncio.wait_for(
                    loop.run_in_executor(
                        self._executor, lambda: pw.poll("/api/powerwalls")
                    ),
                    timeout=5.0,
                )
                if powerwalls_result and isinstance(powerwalls_result, dict):
                    data.powerwalls = powerwalls_result
                elif powerwalls_result and isinstance(powerwalls_result, str):
                    try:
                        data.powerwalls = json.loads(powerwalls_result)
                    except json.JSONDecodeError:
                        pass
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug(f"Powerwalls not available for {gateway_id}: {e}")

            # Guard against partial TEDAPI follower snapshots replacing a complete
            # multi-Powerwall view for a single poll cycle.
            data = self._preserve_complete_multi_pw_snapshot(gateway_id, data)

            # Update cache
            gateway = self.gateways[gateway_id]

            # Log connection success on first connection or reconnection
            was_offline = not gateway.online
            gateway.online = True
            gateway.last_error = None

            # Reset backoff on success
            previous_failures = self._consecutive_failures.get(gateway_id, 0)
            self._consecutive_failures[gateway_id] = 0
            self._next_poll_time[gateway_id] = 0  # Poll normally next cycle

            # Store successful data for graceful degradation
            self._last_successful_data[gateway_id] = data

            if was_offline:
                logger.info(
                    f"Successfully connected to gateway {gateway_id} ({gateway.host})"
                )
                if previous_failures > 0:
                    logger.debug(
                        f"Exponential backoff reset for {gateway_id} after {previous_failures} failures"
                    )

            self.cache[gateway_id] = GatewayStatus(
                gateway=gateway, data=data, online=True, last_updated=data.timestamp
            )

            # Fire-and-forget MQTT publish after the cache is updated.
            # Importing here (late import) avoids a circular dependency at module
            # load time (publisher.py → config → gateway_manager).
            # create_task() ensures MQTT failures never raise into this poll path.
            from app.mqtt.publisher import mqtt_publisher
            if mqtt_publisher.enabled:
                asyncio.create_task(
                    mqtt_publisher.publish_gateway(gateway_id, self.cache[gateway_id]),
                    name=f"mqtt-publish-{gateway_id}",
                )

        except Exception as e:
            gateway = self.gateways[gateway_id]

            # Increment failure count and calculate exponential backoff
            self._consecutive_failures[gateway_id] = (
                self._consecutive_failures.get(gateway_id, 0) + 1
            )
            failure_count = self._consecutive_failures[gateway_id]

            # Exponential backoff: 5s, 10s, 30s, 60s, 120s (max 2 minutes)
            backoff_intervals = [5, 10, 30, 60, 120]
            backoff_index = min(failure_count - 1, len(backoff_intervals) - 1)
            backoff_seconds = backoff_intervals[backoff_index]

            now = datetime.now().timestamp()
            self._next_poll_time[gateway_id] = now + backoff_seconds

            logger.debug(
                f"Exponential backoff for {gateway_id}: failure #{failure_count}, waiting {backoff_seconds}s before retry"
            )

            # Log connection failures with full context
            if gateway.online:
                # Just went offline
                logger.error(
                    f"Lost connection to gateway {gateway_id} ({gateway.host}): {e}"
                )
                logger.info(
                    f"Will retry gateway {gateway_id} in {backoff_seconds}s (failure #{failure_count})"
                )
            else:
                # Still offline, attempting to reconnect
                logger.warning(
                    f"Unable to connect to gateway {gateway_id} ({gateway.host}): {e} - backoff {backoff_seconds}s (failure #{failure_count})"
                )

            gateway.online = False
            gateway.last_error = str(e)

            self.cache[gateway_id] = GatewayStatus(
                gateway=gateway, online=False, error=str(e), last_updated=now
            )

            # Publish the offline status to MQTT so HA reflects gateway going offline.
            from app.mqtt.publisher import mqtt_publisher
            if mqtt_publisher.enabled:
                asyncio.create_task(
                    mqtt_publisher.publish_gateway(gateway_id, self.cache[gateway_id]),
                    name=f"mqtt-publish-{gateway_id}",
                )

    def get_gateway(self, gateway_id: str) -> Optional[GatewayStatus]:
        """Get status for a specific gateway with graceful degradation support.

        Graceful Degradation (PW_GRACEFUL_DEGRADATION=yes):
            - If gateway is offline but went offline recently (within PW_CACHE_TTL seconds)
            - Return cached data with last_updated timestamp
            - After PW_CACHE_TTL expires, return status with data=None

        This allows UI to remain responsive during brief network outages while
        indicating stale data, and eventually showing "offline" after extended downtime.
        """
        from app.config import settings

        status = self.cache.get(gateway_id)
        if not status:
            return None

        # If gateway is online, return current status
        if status.online:
            return status

        # Gateway is offline - check graceful degradation settings
        if not settings.graceful_degradation:
            # Graceful degradation disabled - return offline status with no data
            logger.debug(
                f"Gateway {gateway_id} offline, graceful degradation disabled (PW_GRACEFUL_DEGRADATION=no)"
            )
            return status

        # Check if we have cached data that's still fresh
        last_success_data = self._last_successful_data.get(gateway_id)
        if not last_success_data or not last_success_data.timestamp:
            # No cached data available
            logger.debug(
                f"Gateway {gateway_id} offline, no cached data available for graceful degradation"
            )
            return status

        # Calculate age of cached data
        now = datetime.now().timestamp()
        data_age = now - last_success_data.timestamp

        # If cached data is within TTL, return it with offline status
        if data_age <= settings.cache_ttl:
            logger.debug(
                f"Graceful degradation active for {gateway_id}: serving stale data (age: {data_age:.0f}s / TTL: {settings.cache_ttl}s)"
            )
            return GatewayStatus(
                gateway=status.gateway,
                data=last_success_data,  # Return last good data
                online=False,  # Still indicate gateway is offline
                last_updated=last_success_data.timestamp,
                error=status.error,
            )

        # Cached data too old - return offline status with no data
        logger.debug(
            f"Graceful degradation expired for {gateway_id}: cached data too old (age: {data_age:.0f}s > TTL: {settings.cache_ttl}s), returning null"
        )
        return status

    def get_all_gateways(self) -> Dict[str, GatewayStatus]:
        """Get status for all gateways with graceful degradation applied."""
        result = {}
        for gateway_id in self.gateways.keys():
            status = self.get_gateway(
                gateway_id
            )  # Use get_gateway for graceful degradation
            if status:
                result[gateway_id] = status
        return result

    def get_connection(self, gateway_id: str) -> Optional[pypowerwall.Powerwall]:
        """Get pypowerwall connection for a gateway."""
        return self.connections.get(gateway_id)

    async def call_api(
        self,
        gateway_id: str,
        method: str,
        *args,
        timeout: float = 5.0,
        fail_if_offline: bool = True,
        **kwargs,
    ) -> Optional[Any]:
        """Safely call a pypowerwall API method with timeout protection.

        This wraps blocking pypowerwall calls in the dedicated executor to prevent
        blocking the FastAPI event loop. All direct pypowerwall calls from API
        endpoints should use this method.

        Fast-Fail Behavior:
            By default, returns None immediately if gateway is offline (fail_if_offline=True).
            This prevents wasting time on connections that will likely fail.
            Set fail_if_offline=False for operations that should attempt connection
            regardless of cached status (e.g., reconnection attempts).

        Args:
            gateway_id: Gateway identifier
            method: Method name to call on pypowerwall object (e.g., 'grid_status', 'get_reserve')
            *args: Positional arguments to pass to method
            timeout: Timeout in seconds (default: 5.0)
            fail_if_offline: Return None immediately if gateway offline (default: True)
            **kwargs: Keyword arguments to pass to method

        Returns:
            Result of the pypowerwall method call, or None on error/timeout/offline

        Example:
            grid_status = await gateway_manager.call_api('default', 'grid_status', timeout=3.0)
            reserve = await gateway_manager.call_api('default', 'get_reserve')
        """
        # Fast-fail if gateway is offline
        if fail_if_offline:
            status = self.cache.get(gateway_id)
            if status and not status.online:
                logger.debug(
                    f"[{gateway_id}] call_api({method}) fast-fail: gateway offline"
                )
                return None

        pw = self.connections.get(gateway_id)
        if not pw:
            logger.warning(f"[{gateway_id}] call_api({method}): no connection object")
            return None

        try:
            method_func = getattr(pw, method)
            loop = asyncio.get_running_loop()
            logger.debug(
                f"[{gateway_id}] call_api({method}) starting (timeout={timeout}s)"
            )
            if method in _WRITE_METHODS:
                async with self._write_lock:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(
                            self._executor, lambda: method_func(*args, **kwargs)
                        ),
                        timeout=timeout,
                    )
            else:
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        self._executor, lambda: method_func(*args, **kwargs)
                    ),
                    timeout=timeout,
                )
            logger.debug(f"[{gateway_id}] call_api({method}) completed successfully")
            return result
        except asyncio.TimeoutError:
            logger.warning(
                f"[{gateway_id}] call_api({method}) timeout after {timeout}s"
            )
            return None
        except AttributeError:
            logger.error(f"[{gateway_id}] call_api({method}): method not found")
            return None
        except Exception as e:
            logger.warning(f"[{gateway_id}] call_api({method}) error: {e}")
            return None

    async def cloud_control(
        self, method: str, *args, timeout: float = 10.0, **kwargs
    ) -> Optional[Any]:
        """Call a control method via the cloud connection.

        TEDAPI (local gateway) doesn't support write operations. When cloud
        credentials are configured alongside a TEDAPI gateway, a separate
        cloud-mode pypowerwall connection is created for control operations
        like set_reserve() and set_mode().

        Args:
            method: Method name on pypowerwall (e.g., 'set_reserve', 'set_mode')
            *args: Positional arguments for the method
            timeout: Timeout in seconds (default: 10.0)
            **kwargs: Keyword arguments for the method

        Returns:
            Result of the method call, or None on error/timeout
        """
        if not self._cloud_control:
            logger.error(f"cloud_control({method}): no cloud connection available")
            return None
        try:
            method_func = getattr(self._cloud_control, method)
            loop = asyncio.get_running_loop()
            if method in _WRITE_METHODS:
                async with self._write_lock:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(
                            self._executor, lambda: method_func(*args, **kwargs)
                        ),
                        timeout=timeout,
                    )
            else:
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        self._executor, lambda: method_func(*args, **kwargs)
                    ),
                    timeout=timeout,
                )
            logger.info(f"cloud_control({method}) completed successfully")
            return result
        except asyncio.TimeoutError:
            logger.warning(f"cloud_control({method}) timeout after {timeout}s")
            return None
        except AttributeError:
            logger.error(f"cloud_control({method}): method not found")
            return None
        except Exception as e:
            logger.warning(f"cloud_control({method}) error: {e}")
            return None

    async def call_tedapi(
        self,
        gateway_id: str,
        method: str,
        *args,
        timeout: float = 5.0,
        fail_if_offline: bool = True,
        **kwargs,
    ) -> Optional[Any]:
        """Safely call a TEDAPI method with timeout protection.

        Args:
            gateway_id: Gateway identifier
            method: Method name to call on tedapi object (e.g., 'get_config', 'get_status')
            timeout: Timeout in seconds (default: 5.0)
            fail_if_offline: Return None immediately if gateway offline (default: True)

        Returns:
            Result of the TEDAPI method call, or None if TEDAPI not available/offline
        """
        # Fast-fail if gateway is offline
        if fail_if_offline:
            status = self.cache.get(gateway_id)
            if status and not status.online:
                logger.debug(
                    f"[{gateway_id}] call_tedapi({method}) fast-fail: gateway offline"
                )
                return None

        pw = self.connections.get(gateway_id)
        if not pw or not hasattr(pw, "tedapi") or not pw.tedapi:
            logger.debug(f"[{gateway_id}] call_tedapi({method}): TEDAPI not available")
            return None

        try:
            method_func = getattr(pw.tedapi, method)
            loop = asyncio.get_running_loop()
            logger.debug(
                f"[{gateway_id}] call_tedapi({method}) starting (timeout={timeout}s)"
            )
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor, lambda: method_func(*args, **kwargs)
                ),
                timeout=timeout,
            )
            logger.debug(f"[{gateway_id}] call_tedapi({method}) completed successfully")
            return result
        except asyncio.TimeoutError:
            logger.warning(
                f"[{gateway_id}] call_tedapi({method}) timeout after {timeout}s"
            )
            return None
        except AttributeError:
            logger.error(f"[{gateway_id}] call_tedapi({method}): method not found")
            return None
        except Exception as e:
            logger.warning(f"[{gateway_id}] call_tedapi({method}) error: {e}")
            return None

    def get_aggregate_data(self) -> AggregateData:
        """Get aggregated data from all gateways.

        SMART AGGREGATION NOTES:
        This is a first-pass implementation that will need tuning as we get real-world
        multi-gateway deployments. Current approach:

        - Battery %: Simple average (TODO: weight by capacity when available)
        - Power flows: Simple sum (works for most cases)
        - Grid power: Calculated as site - solar

        Future considerations:
        - Different aggregation strategies per metric type
        - Weighted averages based on system capacity
        - Handling mixed local/cloud gateways
        - Time synchronization across gateways
        - Outlier detection and handling
        """
        aggregate = AggregateData(timestamp=datetime.now().timestamp())

        for gateway_id, status in self.cache.items():
            aggregate.num_gateways += 1

            if not status.online or not status.data:
                continue

            aggregate.num_online += 1
            data = status.data

            # Aggregate battery percentage
            # TODO: Weight by capacity when battery capacity info is available
            if data.soe_raw is not None:
                aggregate.total_battery_percent_raw += data.soe_raw
            if data.soe is not None:
                aggregate.total_battery_percent += data.soe

            # Aggregate power flows (simple sum - works well for separate systems)
            if data.aggregates:
                site = data.aggregates.get("site", {})
                battery = data.aggregates.get("battery", {})
                load = data.aggregates.get("load", {})
                solar = data.aggregates.get("solar", {})

                site_power = site.get("instant_power", 0)
                battery_power = battery.get("instant_power", 0)
                load_power = load.get("instant_power", 0)
                solar_power = solar.get("instant_power", 0)

                logger.debug(
                    f"Gateway {gateway_id} power: site={site_power}, battery={battery_power}, load={load_power}, solar={solar_power}"
                )

                aggregate.total_site_power += site_power
                aggregate.total_battery_power += battery_power
                aggregate.total_load_power += load_power
                aggregate.total_solar_power += solar_power

            aggregate.gateways[gateway_id] = status

        # Calculate average battery percentage (simple average for now)
        if aggregate.num_online > 0:
            aggregate.total_battery_percent_raw /= aggregate.num_online
            aggregate.total_battery_percent /= aggregate.num_online

        # Grid power is the site power (positive = importing, negative = exporting)
        # The "site" meter in aggregates measures grid interaction directly
        aggregate.total_grid_power = aggregate.total_site_power

        # Get grid status from default gateway if available
        default_gateway = self.cache.get("default")
        if default_gateway and default_gateway.data:
            aggregate.grid_status = default_gateway.data.grid_status

        return aggregate


# Global gateway manager instance
gateway_manager = GatewayManager()

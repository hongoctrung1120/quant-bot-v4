"""Portfolio-level risk monitoring and enforcement."""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from core.config import CapitalConfig, RiskConfig
from core.models.order import OrderIntent, OrderSide
from core.models.risk import RiskState
from core.models.signal import Signal
from core.risk.circuit_breaker import CircuitBreaker
from utils.math import max_drawdown
logger=logging.getLogger(__name__)
@dataclass
class PortfolioSnapshot:
    total_equity:float; gross_exposure:float=0.0; net_exposure:float=0.0; asset_exposure:dict[str,float]=field(default_factory=dict); strategy_exposure:dict[str,float]=field(default_factory=dict); leverage:float=0.0; daily_pnl:float=0.0; high_water_mark:float=0.0
class PortfolioRiskMonitor:
    def __init__(self,risk_config:RiskConfig,capital_config:CapitalConfig)->None:
        self._risk_config=risk_config; self._capital_config=capital_config; self._circuit_breaker=CircuitBreaker(risk_config); self._snapshot=PortfolioSnapshot(0.0); self._equity_history=[]
    @property
    def circuit_breaker(self): return self._circuit_breaker
    @property
    def snapshot(self): return self._snapshot
    def reset_session(self,start_equity:float)->None:
        self._circuit_breaker.reset_session(start_equity); self._snapshot=PortfolioSnapshot(start_equity,high_water_mark=start_equity); self._equity_history=[start_equity]
    def update_equity(self,equity:float,timestamp:datetime)->RiskState:
        self._snapshot.total_equity=equity; self._equity_history.append(equity); self._snapshot.high_water_mark=max(self._snapshot.high_water_mark,equity)
        start=self._circuit_breaker.start_equity
        if start is not None: self._snapshot.daily_pnl=equity-start
        return self._circuit_breaker.update(equity,timestamp)
    def update_exposure(self,gross:float,net:float,asset_exposure:Optional[dict[str,float]]=None,strategy_exposure:Optional[dict[str,float]]=None,net_asset_exposure:Optional[dict[str,float]]=None)->None:
        self._snapshot.gross_exposure=gross; self._snapshot.net_exposure=net; self._snapshot.asset_exposure=asset_exposure or {}; self._snapshot.net_asset_exposure=net_asset_exposure or {}; self._snapshot.strategy_exposure=strategy_exposure or {}; self._snapshot.leverage=gross/self._snapshot.total_equity if self._snapshot.total_equity>0 else 0.0
    def check_signal(self,signal:Signal)->bool: return self._circuit_breaker.new_orders_allowed
    def check_order_intent(self,intent:OrderIntent)->bool:
        if not self._circuit_breaker.new_orders_allowed or self._snapshot.total_equity<=0: return False
        price=intent.limit_price or intent.entry_reference if hasattr(intent,'entry_reference') else intent.limit_price
        if not price or price<=0: return False
        notional=abs(intent.quantity)*price; equity=self._snapshot.total_equity
        signed=notional if intent.side==OrderSide.BUY else -notional
        current_asset=self._snapshot.asset_exposure.get(intent.symbol,0.0)
        current_signed_asset=self._snapshot.net_asset_exposure.get(intent.symbol,0.0)
        current_position_notional = current_asset
        # Project the symbol's signed exposure first, then take its absolute value.
        # This correctly handles adding to, reducing, and reversing long/short positions.
        new_signed_asset=current_signed_asset + signed
        new_asset=abs(new_signed_asset)
        projected_gross=self._snapshot.gross_exposure-current_position_notional+new_asset
        if projected_gross/equity*100.0>self._capital_config.max_gross_exposure_pct+1e-9: return False
        projected_net=self._snapshot.net_exposure+signed
        if abs(projected_net)/equity*100.0>self._capital_config.max_net_exposure_pct+1e-9: return False
        if new_asset/equity*100.0>self._capital_config.max_asset_allocation_pct+1e-9: return False
        current_strategy=self._snapshot.strategy_exposure.get(intent.strategy_id,0.0)
        if (current_strategy+notional)/equity*100.0>self._capital_config.max_strategy_allocation_pct+1e-9: return False
        projected_leverage=projected_gross/equity if equity>0 else float('inf')
        if projected_leverage>self._risk_config.max_leverage+1e-9: return False
        return True
    def get_risk_state(self)->str: return self._circuit_breaker.state.value
    def get_max_drawdown(self)->float: return max_drawdown(self._equity_history)

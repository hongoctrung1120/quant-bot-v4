"""Portfolio position tracking with realized/unrealized PnL."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
@dataclass
class Position:
    symbol:str; quantity:float; avg_entry_price:float; side:str; strategy_id:str; opened_at:datetime; unrealized_pnl:float=0.0; realized_pnl:float=0.0
    @property
    def is_long(self)->bool: return self.quantity>0
    @property
    def notional(self)->float: return abs(self.quantity)*self.avg_entry_price
@dataclass
class PositionBook:
    positions:dict[str,Position]=field(default_factory=dict)
    def get(self,symbol:str)->Optional[Position]: return self.positions.get(symbol)
    def update_from_fill(self,symbol:str,side:str,quantity:float,price:float,strategy_id:str,timestamp:datetime)->float:
        signed=quantity if side.upper()=="BUY" else -quantity; pos=self.positions.get(symbol)
        if pos is None:
            self.positions[symbol]=Position(symbol,signed,price,"long" if signed>0 else "short",strategy_id,timestamp); return 0.0
        old=pos.quantity; same=(old>0 and signed>0) or (old<0 and signed<0)
        if same:
            total=abs(old)+abs(signed); pos.avg_entry_price=(abs(old)*pos.avg_entry_price+abs(signed)*price)/total; pos.quantity=old+signed; return 0.0
        close=min(abs(old),abs(signed)); realized=(price-pos.avg_entry_price)*close if old>0 else (pos.avg_entry_price-price)*close
        new=old+signed
        if abs(new)<1e-12: self.positions.pop(symbol,None)
        elif (old>0 and new>0) or (old<0 and new<0): pos.quantity=new; pos.realized_pnl+=realized
        else: self.positions[symbol]=Position(symbol,new,price,"long" if new>0 else "short",strategy_id,timestamp,realized_pnl=realized)
        return realized
    def gross_exposure(self,prices:Optional[dict[str,float]]=None)->float:
        prices=prices or {}; return sum(abs(p.quantity)*prices.get(s,p.avg_entry_price) for s,p in self.positions.items())
    def net_exposure(self,prices:Optional[dict[str,float]]=None)->float:
        prices=prices or {}; return sum(p.quantity*prices.get(s,p.avg_entry_price) for s,p in self.positions.items())
    def asset_exposure(self,prices:Optional[dict[str,float]]=None)->dict[str,float]:
        prices=prices or {}; return {s:abs(p.quantity)*prices.get(s,p.avg_entry_price) for s,p in self.positions.items()}
    def net_asset_exposure(self,prices:Optional[dict[str,float]]=None)->dict[str,float]:
        prices=prices or {}; return {s:p.quantity*prices.get(s,p.avg_entry_price) for s,p in self.positions.items()}
    def strategy_exposure(self,prices:Optional[dict[str,float]]=None)->dict[str,float]:
        prices=prices or {}; out={}
        for s,p in self.positions.items(): out[p.strategy_id]=out.get(p.strategy_id,0.0)+abs(p.quantity)*prices.get(s,p.avg_entry_price)
        return out
    def mark_to_market(self,prices:dict[str,float])->float:
        total=0.0
        for symbol,pos in self.positions.items():
            price=prices.get(symbol,pos.avg_entry_price); pos.unrealized_pnl=(price-pos.avg_entry_price)*pos.quantity if pos.quantity>0 else (pos.avg_entry_price-price)*abs(pos.quantity); total+=pos.unrealized_pnl
        return total

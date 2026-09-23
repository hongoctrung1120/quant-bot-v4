# Tham chiếu toán học của dự án

Tài liệu này liệt kê **toàn bộ** công thức toán/thống kê dùng trong hệ thống,
trích trực tiếp từ code (không suy diễn). Mọi con số ở đây đều **tất định
(deterministic)** — không có machine learning nào trong runtime, dù
`scikit-learn`, `lightgbm`, `hmmlearn` có trong `requirements.txt` (chưa được
gọi ở đâu cả).

Ký hiệu: `close_t` = giá đóng cửa nến hiện tại, `N` = period, `Σ` = tổng.

---

## 1. Indicator / Feature — `core/features/`

| Feature | File | Công thức |
|---|---|---|
| Log return | `price.py` | `ln(close_t / close_{t-1})` |
| Rolling return(N) | `price.py` | `(close_t / close_{t-N}) − 1` |
| EMA(N) | `price.py` | `α = 2/(N+1)`; `EMA_t = α·close_t + (1−α)·EMA_{t-1}`; khởi tạo `EMA_0 = close_0` |
| SMA(N) | `price.py` | Trung bình cộng N giá đóng cửa gần nhất |
| ATR(N) | `volatility.py` | `TR_t = max(high−low, \|high−close_{t-1}\|, \|low−close_{t-1}\|)`; `ATR = mean(TR, N)` (trung bình cộng đơn giản, **không** phải Wilder smoothing) |
| Realized Volatility(N) | `volatility.py` | Độ lệch chuẩn **mẫu** (chia N−1) của log-return trong N nến gần nhất |
| Bollinger Band Width(N, k=2) | `volatility.py` | `upper/lower = mean ± k·std(close,N)`; `width = (upper−lower)/mean` |
| RSI(N) | `momentum.py` | `avg_gain, avg_loss` = trung bình cộng đơn giản N thay đổi giá gần nhất (**không** Wilder EMA); `RS = avg_gain/avg_loss`; `RSI = 100 − 100/(1+RS)`; nếu `avg_loss = 0 → RSI = 100` |
| ROC(N) | `momentum.py` | `((close_t − close_{t-N}) / close_{t-N}) × 100` |
| MACD | `momentum.py` | `EMA(12) − EMA(26)` |
| Relative Volume(N) | `volume.py` | `volume_t / SMA(volume, N)` |
| Dollar Volume SMA(N) | `volume.py` | `SMA(price × volume, N)` |
| VWAP Deviation | `orderflow.py` | `(close − vwap) / vwap` (vwap tính sẵn trong bar) |
| Order Flow Imbalance | `orderflow.py` | `(buy_volume − sell_volume) / (buy_volume + sell_volume)` |
| Rolling Std(N) | `statistical.py` | Độ lệch chuẩn mẫu (chia N−1) của `close` |
| Rolling Skewness(N) | `statistical.py` | `m2 = mean((r−mean)², N)`; `m3 = mean((r−mean)³, N)` (chia N, không N−1); `skew = m3 / m2^1.5` |

Tất cả feature chạy **incremental theo từng bar đã đóng**, tách state riêng
theo symbol — không dùng dữ liệu tương lai (`core/features/feature_engine.py`).

---

## 2. Regime Detection — `core/regime/rules.py`

Cây quyết định theo ngưỡng, đánh giá tuần tự:

```
vol_rank = percentile rank của realized_vol_20 trong 100 giá trị gần nhất

nếu vol_rank ≥ 0.7:            HIGH_VOLATILITY   confidence = 0.6 + 0.3·vol_rank
elif vol_rank ≤ 0.3:           LOW_VOLATILITY    confidence = 0.6 + 0.3·(1−vol_rank)
elif spread = (ema12−ema26)/ema26:
    spread > 0.005:            TREND_BULL        confidence = min(0.95, 0.6+10·|spread|)
    spread < −0.005:           TREND_BEAR        confidence = min(0.95, 0.6+10·|spread|)
    elif 40 ≤ RSI(14) ≤ 60:    MEAN_REVERTING    confidence = 0.65
else:                          TRANSITION        confidence = 0.5
```

---

## 3. Chiến lược entry — Crypto (`core/strategy/`)

| Chiến lược | Điều kiện vào lệnh | Stop / Target |
|---|---|---|
| Trend Following | `ema_12 > ema_26 → BUY`, ngược lại `SELL` | `stop = close ∓ 2·ATR(14)`; `target = close ± 3·\|close−stop\|` (3R) |
| Momentum | `RSI>55 & ROC>0 → BUY`; `RSI<45 & ROC<0 → SELL` | Không có stop riêng — dùng mặc định 2% giá của position sizer |
| Mean Reversion | `vwap_deviation < −z → BUY`; `> +z → SELL` | — |
| Breakout | `close > max(high, 20 nến trước) & relative_volume_20 ≥ 1.5 → BUY` (tương tự low/SELL) | — |

Mỗi chiến lược chỉ hoạt động trong tập regime cho phép (xem code từng file).

---

## 4. Chiến lược entry — Forex (`fx/strategy/`)

**Session Breakout** (`session_breakout.py`):
```
range = high_asia − low_asia   (nến giờ 0h–7h UTC)
điều kiện hợp lệ: 1.5 ≤ range/ATR(14) ≤ 6.0
BUY  khi close > high_asia  (trong cửa sổ giờ 7h–12h UTC, mở phiên London)
SELL khi close < low_asia
stop = mép đối diện của range; target = close ± 1.5·|close−stop|
```

**Trend Pullback đa khung** (`trend_pullback.py`):
```
hướng = dấu của (EMA_50 − EMA_200)/EMA_200   (tính trên khung H4)
LONG:  chờ RSI(14) trên H1 < 45 (armed) → thoát vùng đó → entry
SHORT: chờ RSI(14) trên H1 > 55 (armed) → thoát vùng đó → entry
stop = close ∓ 1.8·ATR(14); target = close ± 2.5·|close−stop|
```

**Currency Strength Cross-sectional** (`currency_strength.py`):
```
momentum_chuẩn_hóa(pair) = rolling_return_20 / (realized_vol_20 · √20)

Với mỗi đồng tiền C: score(C) = trung bình momentum_chuẩn_hóa của mọi cặp
  chứa C (cộng nếu C là base, trừ nếu C là quote)

Xếp hạng score(C) toàn bộ đồng tiền trong universe.
spread = score(mạnh nhất) − score(yếu nhất)
Nếu spread ≥ ngưỡng (mặc định 0.8): trade cặp thể hiện đúng 2 đồng tiền đó
stop = 2.0·ATR(14); target = 2.0R
```

**Trailing stop chung** (`base.py: atr_trailing_stop`):
```
progress_R = (giá hiện tại − entry) / |entry − stop_ban_đầu|   (đổi dấu nếu short)
nếu progress_R ≥ 1.0:
    candidate = close ∓ 2.0~2.5·ATR(14)
    stop mới = candidate CHỈ KHI có lợi hơn stop hiện tại (không bao giờ nới lỏng)
```

---

## 5. Position Sizing

**Crypto** (`core/risk/position_sizing.py`):
```
risk_amount = equity × risk_per_trade_pct
quantity    = risk_amount / stop_distance
quantity    = min(quantity, budget/price, max_position_value/price)
```

**Forex** (`fx/risk/sizing.py`):
```
stop_pips    = |entry − stop| / pip_size
risk_amount  = equity × risk_per_trade_pct × risk_scaler
lots         = floor_to_lot_step(risk_amount / (stop_pips × pip_value_per_lot))

Từ chối lệnh (không tự phóng to) nếu:
  - lots < min_lot của broker
  - vượt trần margin khả dụng
  - vượt trần currency-risk hoặc gross exposure
```

`pip_value_per_lot = pip_size × contract_size`, quy đổi sang account currency
qua `RateBook` (tam giác hóa qua tối đa 1 cặp trung gian).

---

## 6. Risk Governor — `fx/risk/governor.py` (chỉ giảm rủi ro, không bao giờ tăng)

**Drawdown scaler:**
```
dd ≤ warning_drawdown_pct        → 1.0
dd ≥ max_drawdown_pct            → 0.25 (và hệ thống HALT vĩnh viễn)
ở giữa:  scaler = max(1 − 0.75·(dd−warn)/(hard−warn), 0.25)
```

**Volatility scaler:**
```
realized_annual_vol = std(daily_return, ≥20 ngày) × √252
scaler = clamp(target_annual_vol / realized_annual_vol, 0.4, 1.0)
```

**Kelly scaler** (half/quarter-Kelly, cần ≥30 lệnh gần nhất):
```
payoff  = avg_win_R / avg_loss_R
kelly   = win_rate − (1 − win_rate)/payoff
target% = kelly_fraction(0.25) × kelly × 100
scaler  = clamp(target% / risk_per_trade_pct, 0.3, 1.0)
```

**Risk scaler tổng hợp** = `drawdown_scaler × volatility_scaler × kelly_scaler`
(nhân dồn, mỗi thành phần ≤ 1.0 → tích luôn ≤ 1.0).

**Giới hạn lỗ 4 tầng** (không phụ thuộc scaler, chặn cứng):
```
daily_loss_pct   ≥ limit  → khóa đến hôm sau
weekly_loss_pct  ≥ limit  → khóa đến tuần sau
monthly_loss_pct ≥ limit  → khóa đến tháng sau
drawdown_pct     ≥ max    → HALT vĩnh viễn (không tự phục hồi)
```

---

## 7. Currency Exposure / Risk Netting — `fx/risk/exposure.py`

```
Với mỗi vị thế trên cặp base/quote, lots ký hiệu có dấu:
  risk(base)  += risk_amount   (nếu long) hoặc −risk_amount (nếu short)
  risk(quote) += −risk(base)   (ngược dấu, cùng độ lớn)

Chặn lệnh mới nếu: |risk(base) sau khi cộng| > max_currency_risk_pct × equity
                 hoặc tương tự cho quote
```
→ Long EURUSD + Long USDJPY tự động **triệt tiêu** rủi ro USD (một bên short
USD, một bên long USD).

---

## 8. Chi phí giao dịch — `fx/costs.py`

```
spread_pips(t) = typical_spread × spread_stress_multiplier × session_multiplier(t)

fill_price = mid ± (spread_pips/2 + slippage_pips)   [+ khi BUY, − khi SELL]

swap(t) = swap_pips_per_day(long/short) × pip_value_per_lot × lots
          × swap_multiplier(t)     [swap_multiplier = 3 vào thứ Tư, else 1]

commission = |lots| × commission_per_lot_per_side
```

---

## 9. Chỉ số hiệu suất (Performance Metrics)

`backtest/metrics.py` (crypto) và `fx/report.py` (forex) — công thức giống hệt
nhau về bản chất:

```
total_return = (equity_final − equity_initial) / equity_initial

CAGR = (equity_final / equity_initial)^(1/years) − 1
       years = số ngày lịch giữa điểm đầu/cuối / 365.25

annual_vol = std(daily_return) × √252

Sharpe  = mean(daily_return) / std(daily_return) × √252
Sortino = mean(daily_return) / downside_std × √252
          downside_std = √(mean(r² với r<0))     [bán phương sai quanh 0]

Max Drawdown = max( (peak_t − equity_t) / peak_t )  chạy dọc equity curve
               peak_t = max(equity_0..t)

Calmar = CAGR / Max Drawdown

Expectancy(R) = mean(net_pnl / risk_amount)  trên toàn bộ lệnh đã đóng
Profit Factor = Σ(lãi các lệnh thắng) / |Σ(lỗ các lệnh thua)|
```

---

## 10. Thống kê kiểm chứng — `fx/validation/`

**Bootstrap expectancy** (`monte_carlo.py`):
```
Resample có hoàn lại N lần từ tập R-multiple của các lệnh đã đóng
(iterations = 10,000). Với mỗi lần resample, tính mean.
95% CI = [percentile 2.5%, percentile 97.5%] của phân phối mean đó.
significant = True nếu CI hoàn toàn > 0 (không chỉ mean > 0)
```

**Ruin simulation** (`monte_carlo.py`):
```
Resample thứ tự chuỗi R-multiple (paths × horizon lần), compound:
  equity_t = equity_{t-1} × (1 + R_t × risk_per_trade_pct)
Đo qua toàn bộ path: P(max_drawdown ≥ ngưỡng cháy tài khoản)
```

**Walk-forward score** (`walkforward.py`):
```
score = expectancy_R × √(số lệnh trong mẫu)
```
(giống dạng t-statistic — phạt các mẫu ít lệnh, tránh chọn tham số "may mắn"
trên vài lệnh hiếm hoi.) Tối ưu trên tập train, chỉ báo cáo kết quả trên tập
test chưa từng thấy (out-of-sample).

**Synthetic market cho null-hypothesis test** (`fx/data/synthetic.py`):
```
Mỗi đồng tiền: path = cumsum(shock Gaussian), shock_t ~ N(0, σ²·dt)
  (cộng thêm drift theo regime 3 trạng thái nếu mode = regime_switching)
Giá cặp (base/quote) = exp(path_base − path_quote)   → không có arbitrage tam giác

High/Low liên tục giữa 2 điểm lấy mẫu (a, b), phương sai bước σ²:
  u ~ Uniform(0,1)
  high = 0.5·(a + b + √((b−a)² − 2σ²·ln(u)))
  low  = 0.5·(a + b − √((b−a)² − 2σ²·ln(u)))
```
Đây là luật phân phối cực trị của **cầu Brownian** (Brownian bridge maximum/
minimum) — công thức xác suất duy nhất trong dự án không phải thống kê mô tả
đơn giản. Mục đích: mô phỏng đúng bản chất liên tục của giá giữa 2 mốc lấy
mẫu, để stop-loss không "thoát" được những lần chạm giá mà đường liên tục
thực sự đã chạm (bug đã phát hiện và sửa trong quá trình xây dựng).

---

## Tổng kết

Không có mô hình thống kê ước lượng tham số từ dữ liệu (không hồi quy, không
MLE, không mạng neural). Toàn bộ hệ thống là **công thức tường minh, tính lại
được bằng tay** — indicator kỹ thuật cổ điển, cây quyết định theo ngưỡng, và
một bộ công cụ thống kê suy luận (bootstrap, walk-forward, Monte Carlo) dùng
để *kiểm chứng* kết quả chứ không phải để *tạo ra* tín hiệu giao dịch.

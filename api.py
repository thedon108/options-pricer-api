from http.server import BaseHTTPRequestHandler
import json
import urllib.parse
import yfinance as yf
import math
from datetime import datetime, date

def black_scholes(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0:
        return max(0, S - K) if option_type == 'call' else max(0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    nd1 = norm_cdf(d1)
    nd2 = norm_cdf(d2)
    nd1_neg = norm_cdf(-d1)
    nd2_neg = norm_cdf(-d2)
    if option_type == 'call':
        price = S * nd1 - K * math.exp(-r * T) * nd2
    else:
        price = K * math.exp(-r * T) * nd2_neg - S * nd1_neg
    delta = nd1 if option_type == 'call' else nd1 - 1
    gamma = norm_pdf(d1) / (S * sigma * math.sqrt(T))
    theta = (-(S * norm_pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * (nd2 if option_type == 'call' else nd2_neg)) / 365
    vega = S * norm_pdf(d1) * math.sqrt(T) / 100
    return {'price': round(price, 4), 'delta': round(delta, 4), 'gamma': round(gamma, 6), 'theta': round(theta, 4), 'vega': round(vega, 4)}

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def norm_pdf(x):
    return math.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)

def get_rfr(exchange):
    # Yahoo Finance tickers for each country's short-term benchmark rate
    yf_rate_tickers = {
        # North America
        'NMS': '^IRX', 'NGM': '^IRX', 'NCM': '^IRX', 'NYQ': '^IRX',
        'TOR': '^CA2YT=RR',    # Toronto (Yahoo returns 'TOR')
        # Europe
        'LSE':  '^UKTYIELD',   # UK 2yr gilt
        'GER':  '^EUR2YT=RR',  # Frankfurt & Xetra (Yahoo returns 'GER' for .DE stocks)
        'CPH':  '^DKGYIELD',   # Denmark
        'STO':  '^SEKGYIELD',  # Sweden
        'OSL':  '^NOKGYIELD',  # Norway
        'HEL':  '^EUR2YT=RR',  # Finland (Eurozone)
        'AMS':  '^EUR2YT=RR',  # Netherlands
        'BRU':  '^EUR2YT=RR',  # Belgium
        'PAR':  '^EUR2YT=RR',  # Paris
        'YHD':  '^EUR2YT=RR',  # Paris alt (some .PA stocks return 'YHD')
        'MCE':  '^EUR2YT=RR',  # Madrid
        'MIL':  '^EUR2YT=RR',  # Milan
        'VSE':  '^EUR2YT=RR',  # Vienna
        'EBS':  '^CHFGYIELD',  # Switzerland (Yahoo returns 'EBS')
        # Asia-Pacific
        'JPX':  '^JPN2YT=RR',  # Japan (Yahoo returns 'JPX' for .T stocks)
        'HKG':  '^HKIBBOR',    # Hong Kong
        'SES':  '^SGXRATE',    # Singapore (Yahoo returns 'SES')
        'ASX':  '^AUDGYIELD',  # Australia
        'NZE':  '^NZDGYIELD',  # New Zealand (Yahoo returns 'NZE')
        'BSE':  '^INRGYIELD',  # India BSE
        'NSI':  '^INRGYIELD',  # India NSE (Yahoo returns 'NSI')
        'KSC':  '^KR2YT=RR',   # South Korea (Yahoo returns 'KSC')
    }
    # Hardcoded fallbacks if live fetch fails
    fallbacks = {
        'LSE': 0.042, 'GER': 0.025, 'CPH': 0.028,
        'STO': 0.025, 'OSL': 0.040, 'EBS': 0.012,
        'SES': 0.030, 'HKG': 0.040, 'JPX': 0.005,
        'ASX': 0.042, 'NZE': 0.050, 'BSE': 0.065, 'NSI': 0.065,
        'KSC': 0.035, 'TOR': 0.037,
    }
    ticker = yf_rate_tickers.get(exchange)
    if ticker:
        try:
            rate = yf.Ticker(ticker).fast_info.last_price
            if rate and rate > 0:
                return rate / 100
        except Exception:
            pass
    fb = fallbacks.get(exchange)
    if fb is not None:
        return fb
    # True catch-all: use US T-bill as last resort
    try:
        return yf.Ticker('^IRX').fast_info.last_price / 100
    except Exception:
        return 0.045

def bs_iv(market_price, S, K, T, r, option_type='call', tol=1e-6, max_iter=100):
    """Bisection solver: back out IV from market price using Black-Scholes."""
    if T <= 0 or market_price <= 0:
        return None
    intrinsic = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    if market_price <= intrinsic:
        return None
    lo, hi = 0.001, 10.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        bs = black_scholes(S, K, T, r, mid, option_type)
        price = bs['price'] if isinstance(bs, dict) else bs
        if abs(price - market_price) < tol:
            return mid
        if price < market_price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

def interpolate_iv(iv1, t1, iv2, t2, t):
    var1 = iv1 ** 2 * t1
    var2 = iv2 ** 2 * t2
    var_t = var1 + (var2 - var1) * (t - t1) / (t2 - t1)
    return math.sqrt(max(var_t, 0) / t) if t > 0 else iv1

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

        try:
            path = parsed.path.rstrip('/')

            if path == '/api/quote':
                ticker = params.get('ticker', [''])[0].upper()
                t = yf.Ticker(ticker)
                try:
                    info = t.fast_info
                    price = info.last_price
                except (KeyError, TypeError, AttributeError):
                    raise ValueError(f"Ticker '{ticker}' not found — please check the symbol")
                if price is None or not isinstance(price, (int, float)):
                    raise ValueError(f"Ticker '{ticker}' not found — please check the symbol")
                result = {
                    'ticker': ticker,
                    'price': round(price, 2),
                    'change': round(price - info.previous_close, 2),
                    'change_pct': round((price - info.previous_close) / info.previous_close * 100, 2),
                    'exchange': info.exchange,
                    'currency': info.currency,
                }
                self.wfile.write(json.dumps(result).encode())

            elif path == '/api/rfr':
                ticker = params.get('ticker', [''])[0].upper()
                t = yf.Ticker(ticker)
                exchange = t.fast_info.exchange
                rate = get_rfr(exchange)
                if rate is None:
                    irx = yf.Ticker('^IRX')
                    rate = irx.fast_info.last_price / 100
                self.wfile.write(json.dumps({'rate': round(rate, 4), 'exchange': exchange}).encode())

            elif path == '/api/expiries':
                ticker = params.get('ticker', [''])[0].upper()
                t = yf.Ticker(ticker)
                expiries = list(t.options)
                self.wfile.write(json.dumps({'expiries': expiries}).encode())

            elif path == '/api/chain':
                ticker = params.get('ticker', [''])[0].upper()
                expiry = params.get('expiry', [''])[0]
                r_chain = float(params.get('r', [0.045])[0])
                t = yf.Ticker(ticker)
                S = t.fast_info.last_price
                T_chain = max((datetime.strptime(expiry, '%Y-%m-%d').date() - date.today()).days / 365.0, 1/365)
                opt = t.option_chain(expiry)

                def enrich(rows, option_type):
                    out = []
                    for _, row in rows.iterrows():
                        K = float(row['strike'])
                        bid = float(row.get('bid', 0) or 0)
                        ask = float(row.get('ask', 0) or 0)
                        last = float(row.get('lastPrice', 0) or 0)
                        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else last
                        yahoo_iv = float(row.get('impliedVolatility', 0) or 0)
                        solved = bs_iv(mid, S, K, T_chain, r_chain, option_type) if mid > 0 else None
                        iv = solved if solved and 0.01 < solved < 5 else (yahoo_iv if yahoo_iv > 0.01 else None)
                        out.append({
                            'strike': K,
                            'lastPrice': round(last, 4),
                            'bid': round(bid, 4),
                            'ask': round(ask, 4),
                            'mid': round(mid, 4),
                            'impliedVolatility': round(iv, 6) if iv else 0,
                            'iv_solved': solved is not None and 0.01 < solved < 5,
                        })
                    return out

                calls = enrich(opt.calls, 'call')
                puts = enrich(opt.puts, 'put')
                self.wfile.write(json.dumps({'calls': calls, 'puts': puts, 'spot': round(S, 2)}).encode())

            elif path == '/api/price':
                ticker = params.get('ticker', [''])[0].upper()
                strike_start = float(params.get('strike_start', [0])[0])
                strike_end = float(params.get('strike_end', [0])[0])
                strike_interval = float(params.get('strike_interval', [5])[0])
                expiry_str = params.get('expiry', [''])[0]
                r = float(params.get('r', [0.045])[0])

                if strike_end <= strike_start:
                    raise ValueError("Strike end must be greater than strike start")
                if strike_interval <= 0:
                    raise ValueError("Strike interval must be positive")

                target = datetime.strptime(expiry_str, '%Y-%m-%d').date()
                today = date.today()
                if target <= today:
                    raise ValueError(f"Expiry date {expiry_str} is in the past — please choose a future date")

                t = yf.Ticker(ticker)
                S = t.fast_info.last_price
                expiries = list(t.options)
                if not expiries:
                    raise ValueError(f"No listed options found for '{ticker}'")
                T = (target - today).days / 365.0

                # Find bracketing expiries
                expiry_dates = [datetime.strptime(e, '%Y-%m-%d').date() for e in expiries]
                lower_exp = None
                upper_exp = None
                for ed in expiry_dates:
                    if ed <= target:
                        lower_exp = ed
                    elif ed > target and upper_exp is None:
                        upper_exp = ed

                if lower_exp is None:
                    lower_exp = expiry_dates[0]
                if upper_exp is None:
                    upper_exp = expiry_dates[-1]
                    lower_exp = expiry_dates[-2] if len(expiry_dates) > 1 else expiry_dates[-1]

                lower_str = lower_exp.strftime('%Y-%m-%d')
                upper_str = upper_exp.strftime('%Y-%m-%d')

                lower_chain = t.option_chain(lower_str)
                upper_chain = t.option_chain(upper_str)

                T1 = max((lower_exp - today).days / 365.0, 1/365)
                T2 = max((upper_exp - today).days / 365.0, 1/365)

                strikes = []
                k = strike_start
                while k <= strike_end + 0.001:
                    strikes.append(round(k, 2))
                    k += strike_interval

                results_calls = []
                results_puts = []

                for K in strikes:
                    # Get IV from lower bracket
                    lc_row = lower_chain.calls[abs(lower_chain.calls['strike'] - K) < strike_interval / 2]
                    lp_row = lower_chain.puts[abs(lower_chain.puts['strike'] - K) < strike_interval / 2]
                    uc_row = upper_chain.calls[abs(upper_chain.calls['strike'] - K) < strike_interval / 2]
                    up_row = upper_chain.puts[abs(upper_chain.puts['strike'] - K) < strike_interval / 2]

                    def get_iv(rows):
                        if len(rows) > 0:
                            iv = rows.iloc[0]['impliedVolatility']
                            if iv > 0.01:
                                return iv, False
                            return 0.3, True  # fallback
                        return 0.3, True  # fallback

                    def get_market_price(rows):
                        if len(rows) > 0:
                            bid = rows.iloc[0].get('bid', 0)
                            ask = rows.iloc[0].get('ask', 0)
                            last = rows.iloc[0].get('lastPrice', 0)
                            if bid > 0 and ask > 0:
                                return round((bid + ask) / 2, 2)
                            return round(last, 2)
                        return None

                    iv1_c, fb1_c = get_iv(lc_row)
                    iv2_c, fb2_c = get_iv(uc_row)
                    iv1_p, fb1_p = get_iv(lp_row)
                    iv2_p, fb2_p = get_iv(up_row)

                    iv_c = interpolate_iv(iv1_c, T1, iv2_c, T2, T)
                    iv_p = interpolate_iv(iv1_p, T1, iv2_p, T2, T)

                    bs_c = black_scholes(S, K, T, r, iv_c, 'call')
                    bs_p = black_scholes(S, K, T, r, iv_p, 'put')

                    mp_c = get_market_price(lc_row)
                    mp_p = get_market_price(lp_row)

                    moneyness = 'ATM' if abs(K - S) / S < 0.02 else ('ITM' if K < S else 'OTM')

                    results_calls.append({
                        'strike': K,
                        'moneyness': moneyness,
                        'bs_price': bs_c['price'],
                        'market_price': mp_c,
                        'diff': round(bs_c['price'] - mp_c, 2) if mp_c else None,
                        'iv': round(iv_c * 100, 2),
                        'iv_lower': round(iv1_c * 100, 2),
                        'iv_upper': round(iv2_c * 100, 2),
                        'iv_fallback': fb1_c or fb2_c,
                        'delta': bs_c['delta'],
                        'gamma': bs_c['gamma'],
                        'theta': bs_c['theta'],
                        'vega': bs_c['vega'],
                    })

                    results_puts.append({
                        'strike': K,
                        'moneyness': moneyness,
                        'bs_price': bs_p['price'],
                        'market_price': mp_p,
                        'diff': round(bs_p['price'] - mp_p, 2) if mp_p else None,
                        'iv': round(iv_p * 100, 2),
                        'iv_lower': round(iv1_p * 100, 2),
                        'iv_upper': round(iv2_p * 100, 2),
                        'iv_fallback': fb1_p or fb2_p,
                        'delta': bs_p['delta'],
                        'gamma': bs_p['gamma'],
                        'theta': bs_p['theta'],
                        'vega': bs_p['vega'],
                    })

                self.wfile.write(json.dumps({
                    'calls': results_calls,
                    'puts': results_puts,
                    'spot': round(S, 2),
                    'T': round(T, 4),
                    'lower_expiry': lower_str,
                    'upper_expiry': upper_str,
                }).encode())

            elif path == '/api/fx':
                from_currency = params.get('currency', ['USD'])[0]
                if from_currency == 'USD':
                    self.wfile.write(json.dumps({'rate': 1.0, 'from': 'USD', 'to': 'USD'}).encode())
                else:
                    base = 'GBP' if from_currency == 'GBp' else from_currency
                    divisor = 100 if from_currency == 'GBp' else 1
                    fallbacks = {
                        'GBP': 1.27, 'EUR': 1.08, 'JPY': 0.0065, 'CHF': 1.10,
                        'AUD': 0.64, 'CAD': 0.73, 'HKD': 0.128, 'SGD': 0.74,
                        'NZD': 0.60, 'SEK': 0.093, 'NOK': 0.092, 'DKK': 0.144,
                        'INR': 0.012, 'KRW': 0.00073,
                    }
                    try:
                        rate = yf.Ticker(f'{base}USD=X').fast_info.last_price
                        if rate and rate > 0:
                            self.wfile.write(json.dumps({'rate': round(rate / divisor, 6), 'from': from_currency, 'to': 'USD'}).encode())
                            return
                    except Exception:
                        pass
                    fb = fallbacks.get(base, 1.0)
                    self.wfile.write(json.dumps({'rate': round(fb / divisor, 6), 'from': from_currency, 'to': 'USD'}).encode())

            elif path == '/api/search':
                q = params.get('q', [''])[0].strip()
                if not q:
                    raise ValueError("Missing query parameter 'q'")
                results = yf.Search(q, max_results=10).quotes
                us_exchanges = {'NMS', 'NGM', 'NCM', 'NYQ', 'NYSEArca', 'NASDAQ', 'NYSE'}
                def exchange_score(r):
                    if r.get('exchange', '') in us_exchanges: return 0
                    if r.get('quoteType', '') == 'EQUITY': return 1
                    return 2
                results = sorted(results, key=exchange_score)
                matches = [
                    {
                        'ticker': r.get('symbol', ''),
                        'name': r.get('longname') or r.get('shortname', ''),
                        'exchange': r.get('exchange', ''),
                        'type': r.get('quoteType', ''),
                    }
                    for r in results if r.get('symbol')
                ]
                self.wfile.write(json.dumps({'results': matches}).encode())

            else:
                self.wfile.write(json.dumps({'error': 'Unknown endpoint'}).encode())

        except Exception as e:
            self.wfile.write(json.dumps({'error': str(e)}).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def log_message(self, format, *args):
        pass

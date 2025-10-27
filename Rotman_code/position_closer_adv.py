import requests
import signal
from time import sleep, time
import sys

# Performance optimizations
sys.setswitchinterval(0.001)  # Reduce thread switching overhead (default is 0.005)

class ApiException(Exception):
    pass

# Signal handler for graceful shutdown
def signal_handler(signum, frame):
    global shutdown
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    shutdown = True

# Set your API key (replace with your actual key from RIT client)
API_KEY = {'X-API-Key': 'SYO6MNUQ'}
shutdown = False

# Cache for API responses (using dict for O(1) lookups)
cache = {
    'positions': {'data': None, 'timestamp': 0.0},
    'orders': {'data': None, 'timestamp': 0.0},
    'tick': {'data': None, 'timestamp': 0.0}
}
CACHE_DURATION = 0.3  # 300ms cache

# Pre-compile commonly used values
FINAL_TICKS = frozenset([295, 296, 297, 298, 299])  # frozenset for O(1) membership testing

def get_tick(session, use_cache=True):
    """Get current tick of the case with caching"""
    if use_cache and time() - cache['tick']['timestamp'] < CACHE_DURATION:
        return cache['tick']['data']
    
    resp = session.get('http://localhost:9999/v1/case')
    if resp.status_code == 401:
        raise ApiException('Invalid API key')
    case = resp.json()
    result = (case['tick'], case['period'])
    
    cache['tick']['data'] = result
    cache['tick']['timestamp'] = time()
    return result

def get_positions(session, use_cache=True):
    """Get current positions for all securities with caching"""
    if use_cache and time() - cache['positions']['timestamp'] < CACHE_DURATION:
        return cache['positions']['data']
    
    resp = session.get('http://localhost:9999/v1/securities')
    if resp.status_code == 401:
        raise ApiException('Invalid API key')
    result = resp.json()
    
    cache['positions']['data'] = result
    cache['positions']['timestamp'] = time()
    return result

def get_open_orders(session, use_cache=True):
    """Get all open orders with caching"""
    if use_cache and time() - cache['orders']['timestamp'] < CACHE_DURATION:
        return cache['orders']['data']
    
    resp = session.get('http://localhost:9999/v1/orders')
    if resp.status_code == 401:
        raise ApiException('Invalid API key')
    orders = resp.json()
    # List comprehension is faster than filter
    result = [order for order in orders if order['status'] == 'OPEN']
    
    cache['orders']['data'] = result
    cache['orders']['timestamp'] = time()
    return result

def invalidate_cache():
    """Invalidate all cached data - optimized"""
    current_time = time()
    cache['positions']['timestamp'] = 0.0
    cache['orders']['timestamp'] = 0.0
    cache['tick']['timestamp'] = 0.0

def cancel_order(session, order_id):
    """Cancel a specific order by ID"""
    resp = session.delete(f'http://localhost:9999/v1/orders/{order_id}')
    if resp.status_code == 401:
        raise ApiException('Invalid API key')
    invalidate_cache()  # Invalidate cache after modification
    return resp.json()

def submit_market_order(session, ticker, quantity, action):
    """Submit a market order to close position"""
    params = {
        'ticker': ticker,
        'type': 'MARKET',
        'quantity': abs(quantity),
        'action': action
    }
    resp = session.post('http://localhost:9999/v1/orders', params=params)
    if resp.status_code == 401:
        raise ApiException('Invalid API key')
    invalidate_cache()  # Invalidate cache after modification
    return resp.json()

def cancel_stale_orders(session, max_age_seconds=30):
    """Cancel orders that have been open for too long without filling"""
    try:
        open_orders = get_open_orders(session)
        current_tick = get_tick(session)[0]
        
        for order in open_orders:
            order_id = order['order_id']
            ticker = order['ticker']
            order_tick = order.get('tick', 0)
            age = current_tick - order_tick
            
            if age >= max_age_seconds:
                print(f"[Stale Order] Canceling {ticker} order {order_id} (age: {age} ticks)")
                try:
                    cancel_order(session, order_id)
                except Exception as e:
                    print(f"Error canceling stale order {order_id}: {e}")
    except Exception as e:
        pass

def cancel_wrong_side_orders(session):
    """Cancel sell orders if short, cancel buy orders if long"""
    try:
        securities = get_positions(session)
        open_orders = get_open_orders(session)
        
        # Build dict of current positions
        positions = {sec['ticker']: sec.get('position', 0) for sec in securities}
        
        for order in open_orders:
            ticker = order['ticker']
            order_id = order['order_id']
            action = order['action']
            position = positions.get(ticker, 0)
            
            # If long and have buy orders, cancel them
            if position > 0 and action == 'BUY':
                print(f"[Wrong Side] Long {ticker}, canceling BUY order {order_id}")
                try:
                    cancel_order(session, order_id)
                except Exception as e:
                    pass
            
            # If short and have sell orders, cancel them
            elif position < 0 and action == 'SELL':
                print(f"[Wrong Side] Short {ticker}, canceling SELL order {order_id}")
                try:
                    cancel_order(session, order_id)
                except Exception as e:
                    pass
    
    except Exception as e:
        pass

def check_and_cancel_flat_position_orders(session):
    """Continuously check for flat positions and cancel their limit orders"""
    try:
        securities = get_positions(session)
        open_orders = get_open_orders(session)
        
        # Build a dict of current positions
        positions = {sec['ticker']: sec.get('position', 0) for sec in securities}
        
        # Check each limit order
        for order in open_orders:
            ticker = order['ticker']
            order_type = order['type']
            order_id = order['order_id']
            
            # If it's a limit order and position is flat (0), cancel it
            if order_type == 'LIMIT' and positions.get(ticker, 0) == 0:
                print(f"[Auto-cancel] Position flat for {ticker}, canceling LIMIT order {order_id}")
                try:
                    cancel_order(session, order_id)
                except Exception as e:
                    print(f"Error canceling order {order_id}: {e}")
    
    except Exception as e:
        pass

def calculate_net_position(ticker, actual_position, open_orders):
    """Calculate net position accounting for pending orders - optimized"""
    pending_buys = 0
    pending_sells = 0
    
    # Single loop instead of two separate sum() calls
    for order in open_orders:
        if order['ticker'] == ticker:
            qty = order['quantity'] - order.get('quantity_filled', 0)
            if order['action'] == 'BUY':
                pending_buys += qty
            elif order['action'] == 'SELL':
                pending_sells += qty
    
    # Net position = actual position + pending buys - pending sells
    return actual_position + pending_buys - pending_sells

def gradual_position_close(session, tick):
    """Gradually close positions over final 5 ticks (295, 296, 297, 298, 299)"""
    try:
        # Only run at specific ticks - use frozenset for O(1) lookup
        if tick not in FINAL_TICKS:
            return
        
        # OPTION A: Disable cache for final 5 ticks - use fresh data
        securities = get_positions(session, use_cache=False)
        open_orders = get_open_orders(session, use_cache=False)
        
        # Calculate percentage to close based on tick
        ticks_remaining = 300 - tick
        close_percentage = 1.0 / ticks_remaining  # Equal distribution
        
        print(f"\n[Gradual Close] Tick {tick}: Closing {close_percentage*100:.0f}% of net positions (FRESH DATA)")
        
        for security in securities:
            ticker = security['ticker']
            actual_position = security.get('position', 0)
            
            if actual_position == 0:
                continue
            
            # Calculate net position (accounting for pending orders)
            net_position = calculate_net_position(ticker, actual_position, open_orders)
            
            # Calculate amount to close
            close_amount = int(net_position * close_percentage)
            
            if abs(close_amount) < 1:
                continue
            
            # Determine action
            if close_amount > 0:
                action = 'SELL'
                print(f"  {ticker}: Net long {net_position}, selling {close_amount}")
            else:
                action = 'BUY'
                print(f"  {ticker}: Net short {net_position}, buying {abs(close_amount)}")
            
            try:
                submit_market_order(session, ticker, close_amount, action)
            except Exception as e:
                print(f"  Error closing {ticker}: {e}")
        
    except Exception as e:
        print(f"Error in gradual close: {e}")

def cancel_all_orders(session):
    """Cancel ALL open orders (not just limit orders)"""
    try:
        open_orders = get_open_orders(session, use_cache=False)
        canceled_count = 0
        
        for order in open_orders:
            order_id = order['order_id']
            ticker = order['ticker']
            
            print(f"Canceling ALL orders: {ticker} order {order_id}")
            try:
                cancel_order(session, order_id)
                canceled_count += 1
            except Exception as e:
                print(f"Error canceling order {order_id}: {e}")
        
        if canceled_count > 0:
            print(f"Canceled {canceled_count} order(s).")
    
    except Exception as e:
        print(f"Error canceling all orders: {e}")

def close_all_positions(session):
    """Close all open positions using market orders and cancel ALL orders"""
    print("\n[Final Closeout] Canceling all open orders...")
    cancel_all_orders(session)
    
    sleep(0.3)
    
    print("\n[Final Closeout] Closing all remaining positions...")
    securities = get_positions(session, use_cache=False)
    
    for security in securities:
        position = security.get('position', 0)
        ticker = security['ticker']
        
        if position != 0:
            # If we have a long position, sell it
            if position > 0:
                action = 'SELL'
                print(f"Closing LONG position: Selling {position} shares of {ticker}")
            # If we have a short position, buy it back
            else:
                action = 'BUY'
                print(f"Closing SHORT position: Buying {abs(position)} shares of {ticker}")
            
            try:
                result = submit_market_order(session, ticker, position, action)
                print(f"Order submitted for {ticker}: {result}")
            except Exception as e:
                print(f"Error closing position for {ticker}: {e}")

def main():
    """Main loop - monitors time and closes positions gradually"""
    with requests.Session() as s:
        s.headers.update(API_KEY)
        # Enable connection pooling for faster repeated requests
        adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10)
        s.mount('http://', adapter)
        
        print("Position closer started with PERFORMANCE OPTIMIZATIONS...")
        print("- Auto-canceling flat position orders")
        print("- Monitoring for stale orders (30+ tick age)")
        print("- Canceling wrong-side orders (buy when long, sell when short)")
        print("- Gradual position closing (ticks 295-299)")
        print("- API call caching for efficiency")
        print("- TURBO MODE: Faster checking + no cache in final 5 ticks")
        print("- Kernel optimizations: reduced thread switching, connection pooling")
        print()
        
        position_closed = False
        check_counter = 0
        
        while not shutdown:
            try:
                tick, period = get_tick(s)
                
                # OPTION B: Increase frequency during final 5 ticks
                if tick >= 295:
                    check_interval = 0.25  # 250ms during critical period
                else:
                    check_interval = 0.5   # 500ms normal operation
                
                if not position_closed:
                    # Run every cycle
                    check_and_cancel_flat_position_orders(s)
                    
                    # Run less frequently to avoid spam (except during final ticks)
                    check_counter += 1
                    if check_counter % 4 == 0 or tick >= 295:  # More frequent during final ticks
                        cancel_stale_orders(s, max_age_seconds=30)
                        cancel_wrong_side_orders(s)
                    
                    # Gradual closing from tick 295-298
                    if tick >= 295 and tick < 299:
                        gradual_position_close(s, tick)
                
                # Final closeout at tick 299
                if tick >= 299 and not position_closed:
                    print(f"\nFinal second detected (Tick: {tick}). Final closeout...")
                    close_all_positions(s)
                    position_closed = True
                    print("All positions closed.")
                    break
                
                # Display current tick
                if tick % 10 == 0:
                    print(f"Current tick: {tick}/300")
                
                sleep(check_interval)  # Dynamic sleep based on tick
                
            except ApiException as e:
                print(f"API Error: {e}")
                break
            except Exception as e:
                print(f"Unexpected error: {e}")
                sleep(1)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    main()
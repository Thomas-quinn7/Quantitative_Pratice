import requests
import signal
from time import sleep, time

class ApiException(Exception):
    pass

# Signal handler for graceful shutdown
def signal_handler(signum, frame):
    global shutdown
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    shutdown = True

# API Configuration
API_KEY = {'X-API-Key': 'SYO6MNUQ'}
BASE_URL = 'http://localhost:9999/v1'
shutdown = False

# ============================================================================
# CORE API FUNCTIONS
# ============================================================================

def get_tick(session):
    """Get current tick"""
    resp = session.get(f'{BASE_URL}/case')
    if resp.ok:
        return resp.json()['tick']
    return None

def get_positions(session):
    """Get all positions"""
    resp = session.get(f'{BASE_URL}/securities')
    if resp.ok:
        return resp.json()
    return []

def get_open_orders(session):
    """Get all open orders"""
    resp = session.get(f'{BASE_URL}/orders')
    if resp.ok:
        orders = resp.json()
        return [o for o in orders if o['status'] == 'OPEN']
    return []

def cancel_order(session, order_id):
    """Cancel a specific order"""
    try:
        session.delete(f'{BASE_URL}/orders/{order_id}')
    except:
        pass

def submit_market_order(session, ticker, quantity, action):
    """Submit a market order"""
    params = {
        'ticker': ticker,
        'type': 'MARKET',
        'quantity': abs(quantity),
        'action': action
    }
    try:
        resp = session.post(f'{BASE_URL}/orders', params=params)
        return resp.ok
    except:
        return False

# ============================================================================
# MONITORING FUNCTIONS
# ============================================================================

def cancel_flat_position_orders(session):
    """Cancel limit orders for flat positions"""
    try:
        securities = get_positions(session)
        open_orders = get_open_orders(session)
        
        positions = {sec['ticker']: sec.get('position', 0) for sec in securities}
        
        for order in open_orders:
            if order['type'] == 'LIMIT' and positions.get(order['ticker'], 0) == 0:
                cancel_order(session, order['order_id'])
    except:
        pass

def cancel_wrong_side_orders(session):
    """Cancel buy orders when long, sell orders when short"""
    try:
        securities = get_positions(session)
        open_orders = get_open_orders(session)
        
        positions = {sec['ticker']: sec.get('position', 0) for sec in securities}
        
        for order in open_orders:
            ticker = order['ticker']
            position = positions.get(ticker, 0)
            
            if position > 0 and order['action'] == 'BUY':
                cancel_order(session, order['order_id'])
            elif position < 0 and order['action'] == 'SELL':
                cancel_order(session, order['order_id'])
    except:
        pass

def cancel_stale_orders(session, max_age=30):
    """Cancel orders older than max_age ticks"""
    try:
        current_tick = get_tick(session)
        if current_tick is None:
            return
        
        open_orders = get_open_orders(session)
        
        for order in open_orders:
            age = current_tick - order.get('tick', current_tick)
            if age >= max_age:
                cancel_order(session, order['order_id'])
    except:
        pass

# ============================================================================
# CLOSEOUT FUNCTION
# ============================================================================

def final_closeout(session):
    """Close all positions in final 2 seconds - MAXIMUM SPEED"""
    MAX_ORDER_SIZE = 25000
    
    # Trade log for debugging
    trade_log = []
    
    print("\n" + "="*60)
    print("FINAL CLOSEOUT INITIATED - FAST MODE")
    print("="*60)
    
    start_time = time()
    
    # Step 1: Cancel all existing orders (FAST)
    print("\n[1/3] Canceling all existing orders...")
    open_orders = get_open_orders(session)
    for order in open_orders:
        cancel_order(session, order['order_id'])
    print(f"      Canceled {len(open_orders)} orders in {time() - start_time:.3f}s")
    
    # Step 2: Get positions and submit market orders (NO DELAYS)
    print("\n[2/3] Submitting market orders (FAST MODE - NO DELAYS)...")
    securities = get_positions(session)
    
    total_orders = 0
    for security in securities:
        ticker = security['ticker']
        position = security.get('position', 0)
        
        if position == 0:
            continue
        
        # Calculate number of orders needed
        remaining = abs(position)
        action = 'SELL' if position > 0 else 'BUY'
        original_position = position
        
        print(f"\n      Processing {ticker}: {position:,} shares")
        
        # Submit orders in 25k chunks - AS FAST AS POSSIBLE
        order_count = 0
        while remaining > 0:
            chunk = min(remaining, MAX_ORDER_SIZE)
            order_time = time() - start_time
            
            success = submit_market_order(session, ticker, chunk, action)
            
            # Log trade details
            trade_log.append({
                'timestamp': order_time,
                'ticker': ticker,
                'action': action,
                'quantity': chunk,
                'success': success,
                'original_position': original_position
            })
            
            if success:
                print(f"        [{order_time:.3f}s] {action} {chunk:,} {ticker}")
                total_orders += 1
                order_count += 1
            else:
                print(f"        [{order_time:.3f}s] FAILED: {action} {chunk:,} {ticker}")
            
            remaining -= chunk
            # NO SLEEP - SUBMIT AS FAST AS POSSIBLE
        
        print(f"      Completed {ticker}: {order_count} orders submitted")
    
    elapsed = time() - start_time
    print(f"\n      Total: {total_orders} orders in {elapsed:.3f}s")
    
    # Step 3: Final cleanup (minimal delay)
    print("\n[3/3] Final cleanup...")
    sleep(0.05)  # Minimal 50ms delay
    
    # Cancel any remaining limit orders on flat positions
    cancel_flat_position_orders(session)
    
    # Cancel everything one more time
    open_orders = get_open_orders(session)
    for order in open_orders:
        cancel_order(session, order['order_id'])
    
    total_elapsed = time() - start_time
    print(f"\n{'='*60}")
    print(f"CLOSEOUT COMPLETE in {total_elapsed:.3f}s")
    print(f"{'='*60}\n")
    
    # Print detailed trade log
    print("\n" + "="*60)
    print("TRADE LOG - DEBUGGING METRICS")
    print("="*60)
    print(f"\nTotal Trades Attempted: {len(trade_log)}")
    print(f"Successful: {sum(1 for t in trade_log if t['success'])}")
    print(f"Failed: {sum(1 for t in trade_log if not t['success'])}")
    print(f"Total Time: {total_elapsed:.3f}s")
    print(f"\n{'Time (s)':<10} {'Ticker':<8} {'Action':<6} {'Quantity':<10} {'Status':<10} {'Orig Position':<15}")
    print("-" * 70)
    
    for trade in trade_log:
        status = "SUCCESS" if trade['success'] else "FAILED"
        print(f"{trade['timestamp']:<10.3f} {trade['ticker']:<8} {trade['action']:<6} {trade['quantity']:<10,} {status:<10} {trade['original_position']:<15,}")
    
    print("\n" + "="*60 + "\n")

# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    with requests.Session() as session:
        session.headers.update(API_KEY)
        
        print("="*60)
        print("ROTMAN POSITION CLOSER - ADVANCED")
        print("="*60)
        print("\nFeatures:")
        print("  • Auto-cancel flat position orders")
        print("  • Cancel wrong-side orders")
        print("  • Cancel stale orders (30+ ticks)")
        print("  • Final closeout at tick 298 (2 seconds remaining)")
        print("  • Orders capped at 25k, auto-split large positions")
        print(f"\n{'='*60}\n")
        
        closed = False
        counter = 0
        
        while not shutdown:
            try:
                tick = get_tick(session)
                
                if tick is None:
                    sleep(1)
                    continue
                
                # Regular monitoring (before closeout)
                if not closed:
                    counter += 1
                    
                    # Every cycle (500ms)
                    if counter % 2 == 0:
                        cancel_flat_position_orders(session)
                    
                    # Every 2 seconds
                    if counter % 4 == 0:
                        cancel_wrong_side_orders(session)
                        cancel_stale_orders(session)
                    
                    # Status update
                    if tick % 10 == 0:
                        print(f"Tick: {tick}/300")
                
                # Trigger closeout at tick 298
                if tick >= 298 and not closed:
                    final_closeout(session)
                    closed = True
                    break
                
                sleep(0.5)
                
            except KeyboardInterrupt:
                print("\n\nShutdown requested by user.")
                break
            except Exception as e:
                print(f"Error: {e}")
                sleep(1)
        
        print("\nScript terminated.")

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    main()
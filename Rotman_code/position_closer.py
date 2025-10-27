import requests
import signal
from time import sleep

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

def get_tick(session):
    """Get current tick of the case"""
    resp = session.get('http://localhost:9999/v1/case')
    if resp.status_code == 401:
        raise ApiException('Invalid API key')
    case = resp.json()
    return case['tick'], case['period']

def get_positions(session):
    """Get current positions for all securities"""
    resp = session.get('http://localhost:9999/v1/securities')
    if resp.status_code == 401:
        raise ApiException('Invalid API key')
    return resp.json()

def get_open_orders(session):
    """Get all open orders"""
    resp = session.get('http://localhost:9999/v1/orders')
    if resp.status_code == 401:
        raise ApiException('Invalid API key')
    orders = resp.json()
    # Filter for open orders only
    return [order for order in orders if order['status'] == 'OPEN']

def cancel_order(session, order_id):
    """Cancel a specific order by ID"""
    resp = session.delete(f'http://localhost:9999/v1/orders/{order_id}')
    if resp.status_code == 401:
        raise ApiException('Invalid API key')
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
    return resp.json()

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
        # Silently handle errors to avoid spam in the main loop
        pass

def cancel_all_limit_orders_for_tickers(session, tickers):
    """Cancel all limit orders for specific tickers"""
    try:
        open_orders = get_open_orders(session)
        canceled_count = 0
        
        for order in open_orders:
            ticker = order['ticker']
            order_type = order['type']
            order_id = order['order_id']
            
            # Cancel all limit orders for the tickers we just closed
            if order_type == 'LIMIT' and ticker in tickers:
                print(f"Canceling LIMIT order {order_id} for {ticker}")
                try:
                    cancel_order(session, order_id)
                    print(f"Successfully canceled order {order_id}")
                    canceled_count += 1
                except Exception as e:
                    print(f"Error canceling order {order_id}: {e}")
        
        if canceled_count == 0:
            print("No limit orders found to cancel.")
        else:
            print(f"Canceled {canceled_count} limit order(s).")
    
    except Exception as e:
        print(f"Error retrieving or canceling orders: {e}")

def close_all_positions(session):
    """Close all open positions using market orders and cancel all limit orders"""
    securities = get_positions(session)
    tickers_with_positions = []
    
    for security in securities:
        position = security.get('position', 0)
        ticker = security['ticker']
        
        if position != 0:
            tickers_with_positions.append(ticker)
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
    
    # Give a moment for market orders to fill
    sleep(0.5)
    
    # After closing positions, cancel ALL limit orders for the tickers we just closed
    print("\nCanceling all limit orders for closed positions...")
    cancel_all_limit_orders_for_tickers(session, tickers_with_positions)

def main():
    """Main loop - monitors time and closes positions in final second"""
    with requests.Session() as s:
        s.headers.update(API_KEY)
        
        print("Position closer started. Monitoring for final second and flat positions...")
        position_closed = False
        
        while not shutdown:
            try:
                tick, period = get_tick(s)
                
                # Continuously check for flat positions and cancel their limit orders
                if not position_closed:
                    check_and_cancel_flat_position_orders(s)
                
                # Close positions when tick reaches 299 (final second before 300)
                # Adjust this threshold based on your case settings
                if tick >= 299 and not position_closed:
                    print(f"\nFinal second detected (Tick: {tick}). Closing all positions...")
                    close_all_positions(s)
                    position_closed = True
                    print("All positions closed.")
                    break
                
                # Display current tick
                if tick % 10 == 0:  # Print every 10 ticks to avoid spam
                    print(f"Current tick: {tick}/300")
                
                sleep(0.5)  # Check every 500ms
                
            except ApiException as e:
                print(f"API Error: {e}")
                break
            except Exception as e:
                print(f"Unexpected error: {e}")
                sleep(1)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    main()
    
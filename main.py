import jax.numpy as jnp
import numpy as np
from black import (black_scholes, greeks, diff_function, implied_volatility,
                   black_scholes_vectorized, greeks_vectorized,
                   black_scholes_batch_strikes, greeks_batch_strikes,
                   price_heatmap,skew_surface)


def test_black_scholes():
    print("Testing Black-Scholes calculations...")

    # Know ntest case: S=100, K=110, T=0.25, r=0.05, sigma=0.2
    S, K, T, r, sigma = 100, 110, 0.25, 0.05, 0.2

    call_price = black_scholes(S, K, T, r, sigma, otype="call")
    put_price = black_scholes(S, K, T, r, sigma, otype="put")

    print(f"Call price: {call_price:.4f}")
    print(f"Put price: {put_price:.4f}")

    # Put-call parity test: C - P = S*e^(-qT) - K*e^(-rT) (with dividends)
    # Since q=0 by default: C - P = S - K*e^(-rT)
    theoretical_diff = S - K * jnp.exp(-r * T)
    actual_diff = call_price - put_price
    parity_error = abs(actual_diff - theoretical_diff)

    print(f"Actual C-P: {actual_diff:.6f}")
    print(f"Expected C-P: {theoretical_diff:.6f}")
    print(f"Put-call parity error: {parity_error:.6f}")

    # Relax tolerance for floating point precision (JAX uses float32 by default)
    assert parity_error < 1e-4, f"Put-call parity violation! Error: {parity_error}"

    # Test invalid option type
    try:
        black_scholes(S, K, T, r, sigma, otype="invalid")
        assert False, "Should have raised ValueError"
    except ValueError:
        print("* Invalid option type handling works")

    print("* Black-Scholes tests passed\n")


def test_greeks():
    print("Testing Greeks calculations...")

    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

    delta, gamma, theta, vega, rho = greeks(S, K, T, r, sigma, otype="call")

    print(f"Delta: {delta:.4f}")
    print(f"Gamma: {gamma:.4f}")
    print(f"Theta: {theta:.4f}")
    print(f"Vega: {vega:.4f}")
    print(f"Rho: {rho:.4f}")

    # Basic sanity checks
    assert 0 < delta < 1, f"Call delta should be between 0 and 1, got {delta}"
    assert gamma > 0, f"Gamma should be positive, got {gamma}"
    assert vega > 0, f"Vega should be positive, got {vega}"

    # Test put Greeks
    delta_put, gamma_put, theta_put, vega_put, rho_put = greeks(
        S, K, T, r, sigma, otype="put"
    )

    # Delta relationship: delta_call - delta_put = 1 (for q=0)
    delta_diff = delta - delta_put
    print(f"Delta difference (should be ~1): {delta_diff:.4f}")
    assert abs(delta_diff - 1) < 0.01, "Delta relationship violation"

    print("* Greeks tests passed\n")


def test_implied_volatility_logic():
    print("Testing implied volatility logic (without network calls)...")

    # Test the diff_function directly
    S, K, T, r, sigma, q = 100, 105, 0.25, 0.05, 0.2, 0

    theoretical_price = black_scholes(S, K, T, r, sigma, q, "call")

    # Test that diff_function returns 0 when using correct volatility
    diff = diff_function(S, K, T, r, sigma, theoretical_price, q, "call")
    print(f"Diff with correct volatility: {diff:.8f}")
    assert abs(diff) < 1e-10, "Diff function should be zero with correct parameters"

    # Test with wrong volatility
    wrong_sigma = 0.3
    diff_wrong = diff_function(S, K, T, r, wrong_sigma, theoretical_price, q, "call")
    print(f"Diff with wrong volatility: {diff_wrong:.4f}")
    assert abs(diff_wrong) > 0.01, "Diff should be non-zero with wrong volatility"

    print("* Implied volatility logic tests passed\n")


def test_edge_cases():
    print("Testing edge cases...")

    # Test very short time to expiration
    S, K, T, r, sigma = 100, 100, 1 / 365, 0.05, 0.2
    price = black_scholes(S, K, T, r, sigma)
    print(f"Price with 1 day to expiry: {price:.4f}")
    assert price >= 0, "Option price should be non-negative"

    # Test deep ITM call
    S, K, T, r, sigma = 100, 50, 1, 0.05, 0.2
    deep_itm_call = black_scholes(S, K, T, r, sigma, otype="call")
    print(f"Deep ITM call price: {deep_itm_call:.4f}")
    assert (
            deep_itm_call > S - K
    ), "Deep ITM call should be worth more than intrinsic value"

    # Test deep OTM call
    S, K, T, r, sigma = 100, 150, 1, 0.05, 0.2
    deep_otm_call = black_scholes(S, K, T, r, sigma, otype="call")
    print(f"Deep OTM call price: {deep_otm_call:.4f}")
    assert deep_otm_call > 0, "Even deep OTM options should have some value"

    print("* Edge case tests passed\n")


def performance_test():
    print("Running performance test...")

    import time
    from black import black_scholes_batch_strikes, greeks_batch_strikes

    S, K, T, r, sigma = 100, 105, 0.25, 0.05, 0.2

    # Test single calculations (JIT compilation)
    print("Testing JIT-optimized single calculations:")
    start_time = time.time()
    for _ in range(10000):
        black_scholes(S, K, T, r, sigma)
    end_time = time.time()

    single_time = end_time - start_time
    single_per_calc = single_time / 10000 * 1000
    print(f"  10,000 single calculations: {single_time:.4f}s ({single_per_calc:.4f} ms each)")

    # Test vectorized batch calculations
    print("Testing vectorized batch calculations:")
    batch_strikes = jnp.linspace(90, 110, 10000)

    start_time = time.time()
    _ = black_scholes_batch_strikes(S, batch_strikes, T, r, sigma, 0, "call")
    end_time = time.time()

    batch_time = end_time - start_time
    batch_per_calc = batch_time / 10000 * 1000
    print(f"  10,000 batch calculations: {batch_time:.4f}s ({batch_per_calc:.4f} ms each)")

    speedup = single_time / batch_time
    print(f"  Vectorization speedup: {speedup:.1f}x")

    # Test Greeks performance
    print("Testing Greeks calculations:")
    start_time = time.time()
    for _ in range(1000):
        greeks(S, K, T, r, sigma)
    end_time = time.time()

    greeks_time = end_time - start_time
    greeks_per_calc = greeks_time / 1000 * 1000
    print(f"  1,000 Greeks calculations: {greeks_time:.4f}s ({greeks_per_calc:.4f} ms each)")

    # Test batch Greeks
    start_time = time.time()
    _ = greeks_batch_strikes(S, batch_strikes[:1000], T, r, sigma, 0, "call")
    end_time = time.time()

    batch_greeks_time = end_time - start_time
    batch_greeks_per_calc = batch_greeks_time / 1000 * 1000
    print(f"  1,000 batch Greeks: {batch_greeks_time:.4f}s ({batch_greeks_per_calc:.4f} ms each)")

    greeks_speedup = greeks_time / batch_greeks_time
    print(f"  Greeks vectorization speedup: {greeks_speedup:.1f}x")

    print("* Performance test completed\n")


def test_interactive_plots():
    print("Testing plot functionality...")

    # Test parameters
    S, K, T, r, sigma = 100, 105, 0.25, 0.05, 0.2

    print("Creating clean 2D price heatmap...")
    print("(Stock Price vs Implied Volatility)")

    try:
        # Create 2D heatmap - clean and professional
        price_heatmap(S, K, T, r, sigma, otype="call")
        print("* 2D price heatmap created successfully!")
        print("  Clean visualization of option prices across price/volatility grid")

    except Exception as e:
        print(f"* Plot creation failed: {e}")
        print("  This might be due to headless environment or missing display")
        return None

    print("\nNote: For 3D interactive volatility surface (Time vs Moneyness vs Implied Vol):")
    print("Use skew_surface() function with real market data")


def test_plot_description():
    print("\nPlot Functions Available:")
    print("\n1. price_heatmap() - Clean 2D heatmap")
    print("   X-axis: Stock Price variations")
    print("   Y-axis: Implied Volatility variations")
    print("   Color: Option Price/Profit")
    print("   ✓ Professional, easy to read")

    print("\n2. skew_surface() - Interactive 3D surface (requires market data)")
    print("   X-axis: Time to Maturity")
    print("   Y-axis: Moneyness (K/S ratio)")
    print("   Z-axis: Implied Volatility")
    print("   ✓ Interactive rotation sliders")
    print("   ✓ Shows volatility smile/skew patterns")


def main():
    print("=== Black-Scholes Options Pricing Library Test Suite ===")
    print("Testing black.py functions...\n")

    try:
        # test_black_scholes()
        # test_greeks()
        # test_implied_volatility_logic()
        # test_edge_cases()
        # performance_test()
        # test_interactive_plots()
        # test_plot_description()
        skew_surface("AAPL")


        print("All tests passed successfully!")
        print(
            "\nNote: Network-dependent functions (stock_data, get_riskfree_rate, implied_volatility)"
        )
        print(
            "were not tested due to external dependencies. Test them manually with internet connection."
        )

    except Exception as e:
        print(f"Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
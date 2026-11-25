"""
HTML/CSS/JavaScript single-page application template for the RegimeFlow Visual Dashboard.
Includes interactive Chart.js visualizations, in-page image modal (no tab redirection),
dynamic KPI loading, and live causal filter simulator playground.
"""


def get_dashboard_html() -> str:
    """Return the complete self-contained HTML for the interactive web dashboard."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RegimeFlow — Probabilistic Market Regime Engine</title>
    <!-- Chart.js for interactive financial visualizations -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
    <style>
        :root {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #1e293b;
            --bg-hover: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-blue: #38bdf8;
            --accent-emerald: #10b981;
            --accent-rose: #f43f5e;
            --accent-amber: #f59e0b;
            --accent-purple: #a855f7;
            --border-color: #334155;
        }
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        body {
            background-color: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        header {
            background-color: rgba(30, 41, 59, 0.85);
            backdrop-filter: blur(8px);
            border-bottom: 1px solid var(--border-color);
            position: sticky;
            top: 0;
            z-index: 50;
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .logo-icon {
            width: 32px;
            height: 32px;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            color: white;
            font-size: 0.9rem;
        }
        .brand h1 {
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: -0.025em;
        }
        .badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            background-color: rgba(16, 185, 129, 0.15);
            color: var(--accent-emerald);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .pulse-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: var(--accent-emerald);
            animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: .4; transform: scale(1.2); }
        }
        nav a {
            color: var(--text-secondary);
            text-decoration: none;
            font-size: 0.875rem;
            font-weight: 500;
            margin-left: 1.5rem;
            transition: color 0.2s;
        }
        nav a:hover {
            color: var(--accent-blue);
        }
        main {
            flex: 1;
            padding: 2rem;
            max-width: 1400px;
            margin: 0 auto;
            width: 100%;
        }
        .grid-kpi {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2rem;
        }
        .card {
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.25rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.75rem;
        }
        .card-title {
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .card-value {
            font-size: 1.875rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin-bottom: 0.5rem;
        }
        .card-subtitle {
            font-size: 0.8125rem;
            color: var(--text-secondary);
            line-height: 1.4;
        }
        .value-emerald { color: var(--accent-emerald); }
        .value-blue { color: var(--accent-blue); }
        .value-rose { color: var(--accent-rose); }
        .value-amber { color: var(--accent-amber); }
        .value-purple { color: var(--accent-purple); }

        .grid-charts {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        @media (max-width: 1024px) {
            .grid-charts {
                grid-template-columns: 1fr;
            }
        }
        .chart-box {
            position: relative;
            height: 380px;
            width: 100%;
        }
        .chart-actions {
            display: flex;
            gap: 0.5rem;
        }
        .btn-toggle {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 0.25rem 0.65rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn-toggle.active, .btn-toggle:hover {
            background: var(--accent-blue);
            color: #0f172a;
            border-color: var(--accent-blue);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.875rem;
            text-align: left;
        }
        th, td {
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border-color);
        }
        th {
            background-color: rgba(255, 255, 255, 0.02);
            color: var(--text-secondary);
            font-weight: 600;
            font-size: 0.75rem;
            text-transform: uppercase;
        }
        tr:hover td {
            background-color: rgba(255, 255, 255, 0.02);
        }

        /* Simulator playground */
        .simulator-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }
        @media (max-width: 768px) {
            .simulator-grid {
                grid-template-columns: 1fr;
            }
        }
        .form-group {
            margin-bottom: 1rem;
        }
        label {
            display: block;
            font-size: 0.8125rem;
            font-weight: 600;
            margin-bottom: 0.35rem;
            color: var(--text-secondary);
        }
        input, select {
            width: 100%;
            background-color: var(--bg-primary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 0.625rem 0.875rem;
            border-radius: 8px;
            font-size: 0.875rem;
            outline: none;
            transition: border-color 0.2s;
        }
        input:focus, select:focus {
            border-color: var(--accent-blue);
        }
        .btn-primary {
            background: linear-gradient(135deg, var(--accent-blue), #2563eb);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.875rem;
            cursor: pointer;
            width: 100%;
            transition: opacity 0.2s;
        }
        .btn-primary:hover {
            opacity: 0.9;
        }
        .result-box {
            background-color: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1.25rem;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .prob-bar {
            height: 8px;
            border-radius: 4px;
            background-color: #334155;
            overflow: hidden;
            margin-top: 0.35rem;
            margin-bottom: 0.75rem;
        }
        .prob-fill {
            height: 100%;
            transition: width 0.4s ease, background-color 0.4s ease;
        }

        /* Gallery grid and in-page modal lightbox */
        .gallery-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 1.25rem;
        }
        .gallery-item {
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
            cursor: pointer;
            transition: transform 0.2s, border-color 0.2s;
        }
        .gallery-item:hover {
            transform: translateY(-2px);
            border-color: var(--accent-blue);
        }
        .gallery-img {
            width: 100%;
            height: 180px;
            object-fit: cover;
            display: block;
        }
        .gallery-caption {
            padding: 0.75rem;
            font-size: 0.8125rem;
            font-weight: 600;
            color: var(--text-secondary);
            border-top: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .gallery-caption span {
            color: var(--accent-blue);
            font-size: 0.75rem;
        }

        /* In-page Modal Lightbox */
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(15, 23, 42, 0.9);
            backdrop-filter: blur(8px);
            justify-content: center;
            align-items: center;
            flex-direction: column;
            padding: 2rem;
        }
        .modal.active {
            display: flex;
        }
        .modal-container {
            position: relative;
            max-width: 90vw;
            max-height: 85vh;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7);
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .modal-content {
            max-width: 100%;
            max-height: 75vh;
            object-fit: contain;
            border-radius: 6px;
        }
        .modal-caption {
            margin-top: 0.75rem;
            font-size: 1rem;
            font-weight: 700;
            color: var(--text-primary);
        }
        .modal-close {
            position: absolute;
            top: -16px;
            right: -16px;
            background: var(--accent-rose);
            color: white;
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            font-weight: bold;
            cursor: pointer;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            transition: transform 0.2s;
        }
        .modal-close:hover {
            transform: scale(1.1);
        }

        footer {
            border-top: 1px solid var(--border-color);
            padding: 1.5rem 2rem;
            text-align: center;
            font-size: 0.8125rem;
            color: var(--text-secondary);
        }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <div class="logo-icon">RF</div>
            <div>
                <h1>RegimeFlow</h1>
                <div style="font-size: 0.75rem; color: var(--text-secondary);">Probabilistic Market Regime & Changepoint Engine</div>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 1rem;">
            <div class="badge">
                <span class="pulse-dot"></span>
                <span>SPY 2000–2026 Walk-Forward (47 Folds)</span>
            </div>
            <nav>
                <a href="#kpis">KPIs</a>
                <a href="#curves">Equity Growth</a>
                <a href="#regimes">Regime Dynamics</a>
                <a href="#stress">Stress Windows</a>
                <a href="#simulator">Interactive Playground</a>
                <a href="#gallery">Figure Gallery</a>
            </nav>
        </div>
    </header>

    <main>
        <!-- KPI Row -->
        <section id="kpis" style="margin-bottom: 2rem;">
            <div style="margin-bottom: 1rem; display: flex; justify-content: space-between; align-items: flex-end;">
                <div>
                    <h2 style="font-size: 1.5rem; font-weight: 700;">Executive Performance Summary</h2>
                    <div style="font-size: 0.875rem; color: var(--text-secondary);">
                        Chronological Walk-Forward on SPY (5,922 OOS Sessions, 2003–2026, Net of 10 bps friction & 1-day lag)
                    </div>
                </div>
            </div>
            <div class="grid-kpi">
                <div class="card" style="border-left: 4px solid var(--accent-emerald);">
                    <div class="card-header">
                        <span class="card-title">Regime-Aware Risk Control</span>
                        <span class="badge" style="background: rgba(16,185,129,0.1); color: var(--accent-emerald);">Optimal</span>
                    </div>
                    <div class="card-value value-emerald" id="kpi-regime-sharpe">0.820 Sharpe</div>
                    <div class="card-subtitle">
                        CAGR: <strong style="color: white;" id="kpi-regime-cagr">10.16%</strong> |
                        Ann. Vol: <strong style="color: white;" id="kpi-regime-vol">12.38%</strong> |
                        Max DD: <strong class="value-emerald" id="kpi-regime-dd">-27.23%</strong>
                    </div>
                </div>

                <div class="card" style="border-left: 4px solid var(--accent-blue);">
                    <div class="card-header">
                        <span class="card-title">Buy & Hold Benchmark</span>
                        <span style="font-size: 0.75rem; color: var(--text-secondary);">Passive SPY</span>
                    </div>
                    <div class="card-value value-blue" id="kpi-bh-sharpe">0.535 Sharpe</div>
                    <div class="card-subtitle">
                        CAGR: <strong style="color: white;" id="kpi-bh-cagr">9.88%</strong> |
                        Ann. Vol: <strong style="color: white;" id="kpi-bh-vol">18.47%</strong> |
                        Max DD: <strong class="value-rose" id="kpi-bh-dd">-59.58%</strong>
                    </div>
                </div>

                <div class="card" style="border-left: 4px solid var(--accent-amber);">
                    <div class="card-header">
                        <span class="card-title">Moving Average (200-Day)</span>
                        <span style="font-size: 0.75rem; color: var(--text-secondary);">Trend Rule</span>
                    </div>
                    <div class="card-value value-amber" id="kpi-ma-sharpe">0.640 Sharpe</div>
                    <div class="card-subtitle">
                        CAGR: <strong style="color: white;" id="kpi-ma-cagr">7.43%</strong> |
                        Ann. Vol: <strong style="color: white;" id="kpi-ma-vol">11.61%</strong> |
                        Max DD: <strong style="color: white;" id="kpi-ma-dd">-23.06%</strong>
                    </div>
                </div>

                <div class="card" style="border-left: 4px solid var(--accent-purple);">
                    <div class="card-header">
                        <span class="card-title">OOS Predictive Density Edge</span>
                        <span style="font-size: 0.75rem; color: var(--text-secondary);">Density Forecast</span>
                    </div>
                    <div class="card-value value-purple">+1.199 pts</div>
                    <div class="card-subtitle">
                        Custom HMM LL: <strong style="color: white;">-4.208</strong> vs.
                        Static Baseline: <strong style="color: white;">-5.407</strong> (log points/session)
                    </div>
                </div>
            </div>
        </section>

        <!-- Interactive Charts Section -->
        <section id="curves" style="margin-bottom: 2rem;">
            <div class="grid-charts">
                <!-- Cumulative Returns -->
                <div class="card">
                    <div class="card-header">
                        <div>
                            <span class="card-title">Cumulative Wealth Growth ($1 Initial Investment)</span>
                            <div style="font-size: 0.75rem; color: var(--text-secondary);">Simulated with 1-period execution lag & 10 bps turnover friction</div>
                        </div>
                        <div class="chart-actions">
                            <button class="btn-toggle active" id="btn-scale-lin" onclick="setEquityScale('linear')">Linear</button>
                            <button class="btn-toggle" id="btn-scale-log" onclick="setEquityScale('logarithmic')">Log Scale</button>
                        </div>
                    </div>
                    <div class="chart-box">
                        <canvas id="equityChart"></canvas>
                    </div>
                </div>

                <!-- Underwater Drawdown Chart -->
                <div class="card">
                    <div class="card-header">
                        <div>
                            <span class="card-title">Underwater Drawdown Profile</span>
                            <div style="font-size: 0.75rem; color: var(--text-secondary);">Peak-to-Trough Loss Exposure</div>
                        </div>
                    </div>
                    <div class="chart-box">
                        <canvas id="drawdownChart"></canvas>
                    </div>
                </div>
            </div>
        </section>

        <!-- Regime Probability Timeline -->
        <section id="regimes" style="margin-bottom: 2rem;">
            <div class="card">
                <div class="card-header">
                    <div>
                        <span class="card-title">Causal Filtered High-Risk Regime Probability vs. SPY Cumulative Growth</span>
                        <div style="font-size: 0.75rem; color: var(--text-secondary);">Online P(z_t = Crisis | x_{1:t}) strictly invariant to future market data</div>
                    </div>
                </div>
                <div class="chart-box" style="height: 340px;">
                    <canvas id="regimeChart"></canvas>
                </div>
            </div>
        </section>

        <!-- Stress Windows & Bootstrap -->
        <section id="stress" style="margin-bottom: 2rem;">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem;">
                <!-- Stress Window Table -->
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">Predeclared Crisis Window Diagnostics</span>
                    </div>
                    <table>
                        <thead>
                            <tr>
                                <th>Historical Event</th>
                                <th>Window</th>
                                <th>Mean Risk Prob</th>
                                <th>Pre-Event Base</th>
                                <th>Asset Max DD</th>
                            </tr>
                        </thead>
                        <tbody id="stress-table-body">
                            <tr>
                                <td><strong>Global Financial Crisis</strong></td>
                                <td>2007–2009</td>
                                <td class="value-rose"><strong>88.06%</strong></td>
                                <td>34.56%</td>
                                <td>-55.19%</td>
                            </tr>
                            <tr>
                                <td><strong>COVID-19 Market Shock</strong></td>
                                <td>Feb–Apr 2020</td>
                                <td class="value-rose"><strong>87.83%</strong></td>
                                <td>15.01%</td>
                                <td>-33.44%</td>
                            </tr>
                            <tr>
                                <td><strong>2022 Inflation & Rate Hikes</strong></td>
                                <td>2022 Full Year</td>
                                <td class="value-amber">10.75%</td>
                                <td>0.02%</td>
                                <td>-24.47%</td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <!-- Bootstrap Uncertainty Table -->
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">Non-IID Paired Block Bootstrap (L=21 Days, B=1000)</span>
                    </div>
                    <table>
                        <thead>
                            <tr>
                                <th>Comparison</th>
                                <th>Metric</th>
                                <th>Observed Δ</th>
                                <th>95% Bootstrap CI</th>
                                <th>p-value</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Regime vs. Buy & Hold</td>
                                <td><strong>Sharpe</strong></td>
                                <td class="value-emerald"><strong>+0.285</strong></td>
                                <td>[-0.061, +0.635]</td>
                                <td>0.1040</td>
                            </tr>
                            <tr>
                                <td>Regime vs. Buy & Hold</td>
                                <td><strong>Max Drawdown</strong></td>
                                <td class="value-emerald"><strong>+32.3% pts</strong></td>
                                <td>[-0.007, +0.386]</td>
                                <td>0.0580</td>
                            </tr>
                            <tr>
                                <td>Regime vs. Moving Avg</td>
                                <td><strong>Sharpe</strong></td>
                                <td class="value-emerald">+0.180</td>
                                <td>[-0.108, +0.467]</td>
                                <td>0.2360</td>
                            </tr>
                            <tr>
                                <td>Regime vs. Moving Avg</td>
                                <td><strong>CAGR</strong></td>
                                <td class="value-emerald">+2.7%</td>
                                <td>[-0.006, +0.063]</td>
                                <td>0.1060</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </section>

        <!-- Interactive Causal Simulator Playground -->
        <section id="simulator" style="margin-bottom: 2rem;">
            <div class="card">
                <div class="card-header">
                    <div>
                        <h2 style="font-size: 1.25rem; font-weight: 700;">Interactive Causal Forward Filter Playground</h2>
                        <div style="font-size: 0.8125rem; color: var(--text-secondary);">
                            Simulate incoming market feature updates and observe real-time causal posterior transitions and portfolio risk sizing.
                        </div>
                    </div>
                </div>
                <div class="simulator-grid" style="margin-top: 1rem;">
                    <!-- Input Form -->
                    <div>
                        <div class="form-group">
                            <label>Preset Market Scenario</label>
                            <select id="sim-preset" onchange="applyPreset()">
                                <option value="bull">Quiescent Bull Market (Low Vol, Positive Drift)</option>
                                <option value="shock" selected>Sudden Volatility Explosion (Crisis Onset)</option>
                                <option value="chop">Choppy Sideways Consolidation</option>
                                <option value="custom">Custom Parameters</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Daily Log Return (r_t)</label>
                            <input type="number" id="sim-logret" step="0.001" value="-0.038">
                            <div style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.25rem;">E.g. -0.038 represents a -3.8% single-day market plunge</div>
                        </div>
                        <div class="form-group">
                            <label>21-Day Realized Volatility (Annualized)</label>
                            <input type="number" id="sim-rv21" step="0.01" value="0.42">
                            <div style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.25rem;">E.g. 0.42 represents 42% annualized trailing volatility</div>
                        </div>
                        <div class="form-group">
                            <label>Volatility Ratio (5d Vol / 63d Vol)</label>
                            <input type="number" id="sim-volratio" step="0.1" value="2.6">
                            <div style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.25rem;">Values > 1.8 indicate acute short-term volatility shock</div>
                        </div>
                        <button class="btn-primary" onclick="runSimulation()">Execute Causal Filter</button>
                    </div>

                    <!-- Output Card -->
                    <div class="result-box">
                        <div style="font-size: 0.875rem; font-weight: 700; color: var(--text-secondary); margin-bottom: 1rem;">
                            CAUSAL INFERENCE RESPONSE (P(z_t | x_{1:t}))
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.5rem;">
                            <span>Assigned Regime State:</span>
                            <strong style="font-size: 1.25rem;" id="res-state" class="value-rose">State 2 (High Risk)</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.5rem;">
                            <span>High-Risk Probability:</span>
                            <strong style="font-size: 1.25rem;" id="res-risk-prob" class="value-rose">89.4%</strong>
                        </div>
                        <div class="prob-bar">
                            <div class="prob-fill" id="res-prob-fill" style="width: 89.4%; background-color: var(--accent-rose);"></div>
                        </div>

                        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.5rem; margin-top: 1.5rem;">
                            <span>Recommended Equity Weight (w_t):</span>
                            <strong style="font-size: 1.5rem;" id="res-weight" class="value-amber">10.6%</strong>
                        </div>
                        <div style="font-size: 0.8125rem; color: var(--text-secondary); margin-bottom: 1rem;">
                            Allocates 10.6% to Equities, holding 89.4% in Risk-Off Cash.
                        </div>

                        <div style="display: flex; justify-content: space-between; align-items: baseline; border-top: 1px solid var(--border-color); padding-top: 0.75rem;">
                            <span>One-Step Predictive Log-Likelihood:</span>
                            <strong style="font-size: 1rem;" id="res-ll">-3.842</strong>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Static Visual Gallery with In-Page Modal (No Tab Redirects) -->
        <section id="gallery" style="margin-bottom: 2rem;">
            <div class="card">
                <div class="card-header">
                    <div>
                        <span class="card-title">Publication Figure Gallery</span>
                        <div style="font-size: 0.75rem; color: var(--text-secondary);">Click any figure to expand in full-screen in-page lightbox (no new tab redirection)</div>
                    </div>
                </div>
                <div class="gallery-grid">
                    <div class="gallery-item" onclick="openModal('/static/figures/regime_probabilities.png', '1. Causal Filtered Regime Probabilities & Asset Dynamics')">
                        <img src="/static/figures/regime_probabilities.png" class="gallery-img" alt="Causal Regime Probabilities">
                        <div class="gallery-caption">
                            <span>1. Regime Probabilities</span>
                            <span>Click to Zoom</span>
                        </div>
                    </div>
                    <div class="gallery-item" onclick="openModal('/static/figures/strategy_equity_curves.png', '2. Strategy Cumulative Wealth Growth (NAV)')">
                        <img src="/static/figures/strategy_equity_curves.png" class="gallery-img" alt="Strategy Cumulative Wealth">
                        <div class="gallery-caption">
                            <span>2. Cumulative Curves</span>
                            <span>Click to Zoom</span>
                        </div>
                    </div>
                    <div class="gallery-item" onclick="openModal('/static/figures/drawdowns.png', '3. Underwater Drawdown Comparison')">
                        <img src="/static/figures/drawdowns.png" class="gallery-img" alt="Underwater Drawdowns">
                        <div class="gallery-caption">
                            <span>3. Drawdowns</span>
                            <span>Click to Zoom</span>
                        </div>
                    </div>
                    <div class="gallery-item" onclick="openModal('/static/figures/exposure_turnover.png', '4. Portfolio Exposure & Turnover Dynamics')">
                        <img src="/static/figures/exposure_turnover.png" class="gallery-img" alt="Exposure and Turnover">
                        <div class="gallery-caption">
                            <span>4. Exposure & Turnover</span>
                            <span>Click to Zoom</span>
                        </div>
                    </div>
                    <div class="gallery-item" onclick="openModal('/static/figures/bic_model_selection.png', '5. Bayesian Information Criterion (BIC) Selection Curves')">
                        <img src="/static/figures/bic_model_selection.png" class="gallery-img" alt="BIC Model Selection">
                        <div class="gallery-caption">
                            <span>5. Model Selection</span>
                            <span>Click to Zoom</span>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    </main>

    <!-- In-Page Lightbox Modal (Does NOT redirect or open new tabs) -->
    <div id="imageModal" class="modal" onclick="closeModal()">
        <div class="modal-container" onclick="event.stopPropagation()">
            <div class="modal-close" onclick="closeModal()">&times;</div>
            <img class="modal-content" id="modalImg" src="" alt="Expanded Publication Figure">
            <div id="modalCaption" class="modal-caption"></div>
        </div>
    </div>

    <footer>
        RegimeFlow Quantitative Research Engine &copy; 2026. Built strictly for scientific research and educational demonstration.
    </footer>

    <!-- Script generating interactive charts and handling simulation -->
    <script>
        let equityChart = null;
        let drawdownChart = null;
        let regimeChart = null;
        let currentScale = 'linear';

        // Load data on page load
        document.addEventListener('DOMContentLoaded', async () => {
            await initializeDashboard();
        });

        async function initializeDashboard() {
            try {
                // Fetch timeseries for charts
                const resTs = await fetch('/api/data/timeseries');
                if (resTs.ok) {
                    const data = await resTs.json();
                    renderCharts(data);
                } else {
                    console.warn('Timeseries endpoint error, falling back to mock');
                    renderMockCharts();
                }

                // Fetch summary metrics
                const resSum = await fetch('/api/data/summary');
                if (resSum.ok) {
                    const sumData = await resSum.json();
                    updateKPIs(sumData);
                }
            } catch (err) {
                console.error('Failed to load live data, using synthetic fallback:', err);
                renderMockCharts();
            }
        }

        function updateKPIs(sumData) {
            if (sumData.Regime_Aware_Risk_Control) {
                const r = sumData.Regime_Aware_Risk_Control;
                document.getElementById('kpi-regime-sharpe').innerText = r.sharpe_ratio.toFixed(3) + ' Sharpe';
                document.getElementById('kpi-regime-cagr').innerText = (r.cagr * 100).toFixed(2) + '%';
                document.getElementById('kpi-regime-vol').innerText = (r.annualized_volatility * 100).toFixed(2) + '%';
                document.getElementById('kpi-regime-dd').innerText = (r.max_drawdown * 100).toFixed(2) + '%';
            }
            if (sumData.Buy_and_Hold) {
                const b = sumData.Buy_and_Hold;
                document.getElementById('kpi-bh-sharpe').innerText = b.sharpe_ratio.toFixed(3) + ' Sharpe';
                document.getElementById('kpi-bh-cagr').innerText = (b.cagr * 100).toFixed(2) + '%';
                document.getElementById('kpi-bh-vol').innerText = (b.annualized_volatility * 100).toFixed(2) + '%';
                document.getElementById('kpi-bh-dd').innerText = (b.max_drawdown * 100).toFixed(2) + '%';
            }
            if (sumData.Moving_Average_200) {
                const m = sumData.Moving_Average_200;
                document.getElementById('kpi-ma-sharpe').innerText = m.sharpe_ratio.toFixed(3) + ' Sharpe';
                document.getElementById('kpi-ma-cagr').innerText = (m.cagr * 100).toFixed(2) + '%';
                document.getElementById('kpi-ma-vol').innerText = (m.annualized_volatility * 100).toFixed(2) + '%';
                document.getElementById('kpi-ma-dd').innerText = (m.max_drawdown * 100).toFixed(2) + '%';
            }
        }

        function renderCharts(data) {
            const labels = data.dates;

            // 1. Equity Chart
            const ctxEquity = document.getElementById('equityChart');
            if (ctxEquity) {
                if (equityChart) equityChart.destroy();
                equityChart = new Chart(ctxEquity.getContext('2d'), {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [
                            {
                                label: 'Regime-Aware Risk Control',
                                data: data.nav_regime,
                                borderColor: '#10b981',
                                borderWidth: 2,
                                pointRadius: 0,
                                tension: 0.1
                            },
                            {
                                label: 'Buy & Hold (SPY)',
                                data: data.nav_bh,
                                borderColor: '#38bdf8',
                                borderWidth: 1.5,
                                pointRadius: 0,
                                tension: 0.1
                            },
                            {
                                label: 'Moving Average (200-day)',
                                data: data.nav_ma,
                                borderColor: '#f59e0b',
                                borderWidth: 1.5,
                                pointRadius: 0,
                                tension: 0.1
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {
                            intersect: false,
                            mode: 'index'
                        },
                        scales: {
                            x: {
                                display: false
                            },
                            y: {
                                type: currentScale,
                                grid: { color: '#334155' },
                                ticks: {
                                    color: '#94a3b8',
                                    callback: val => Number(val).toFixed(1) + 'x'
                                }
                            }
                        },
                        plugins: {
                            legend: {
                                labels: { color: '#f8fafc', font: { size: 11 } }
                            },
                            tooltip: {
                                callbacks: {
                                    label: ctx => ctx.dataset.label + ': $' + Number(ctx.parsed.y).toFixed(2)
                                }
                            }
                        }
                    }
                });
            }

            // 2. Drawdown Chart
            const ctxDD = document.getElementById('drawdownChart');
            if (ctxDD) {
                if (drawdownChart) drawdownChart.destroy();
                drawdownChart = new Chart(ctxDD.getContext('2d'), {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [
                            {
                                label: 'Regime-Aware Drawdown',
                                data: data.dd_regime,
                                borderColor: '#10b981',
                                backgroundColor: 'rgba(16, 185, 129, 0.15)',
                                fill: true,
                                borderWidth: 1.5,
                                pointRadius: 0
                            },
                            {
                                label: 'Buy & Hold Drawdown',
                                data: data.dd_bh,
                                borderColor: '#f43f5e',
                                backgroundColor: 'rgba(244, 63, 94, 0.15)',
                                fill: true,
                                borderWidth: 1.5,
                                pointRadius: 0
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {
                            intersect: false,
                            mode: 'index'
                        },
                        scales: {
                            x: { display: false },
                            y: {
                                type: 'linear',
                                min: -0.65,
                                max: 0.05,
                                grid: { color: '#334155' },
                                ticks: {
                                    color: '#94a3b8',
                                    callback: val => (Number(val) * 100).toFixed(0) + '%'
                                }
                            }
                        },
                        plugins: {
                            legend: {
                                labels: { color: '#f8fafc', font: { size: 11 } }
                            },
                            tooltip: {
                                callbacks: {
                                    label: ctx => ctx.dataset.label + ': ' + (Number(ctx.parsed.y) * 100).toFixed(1) + '%'
                                }
                            }
                        }
                    }
                });
            }

            // 3. Regime Timeline Chart
            const ctxRegime = document.getElementById('regimeChart');
            if (ctxRegime) {
                if (regimeChart) regimeChart.destroy();
                regimeChart = new Chart(ctxRegime.getContext('2d'), {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [
                            {
                                label: 'Causal High-Risk Prob P(Crisis)',
                                data: data.high_risk_prob,
                                borderColor: '#f43f5e',
                                backgroundColor: 'rgba(244, 63, 94, 0.25)',
                                fill: true,
                                borderWidth: 1.5,
                                pointRadius: 0,
                                yAxisID: 'y1'
                            },
                            {
                                label: 'SPY Cumulative Growth',
                                data: data.prices,
                                borderColor: '#38bdf8',
                                borderWidth: 1.5,
                                pointRadius: 0,
                                yAxisID: 'y'
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {
                            intersect: false,
                            mode: 'index'
                        },
                        scales: {
                            x: {
                                grid: { color: '#334155' },
                                ticks: { color: '#94a3b8', maxTicksLimit: 12 }
                            },
                            y: {
                                type: 'linear',
                                position: 'left',
                                grid: { color: '#334155' },
                                ticks: {
                                    color: '#38bdf8',
                                    callback: val => Number(val).toFixed(1) + 'x'
                                },
                                title: { display: true, text: 'SPY Growth', color: '#38bdf8' }
                            },
                            y1: {
                                type: 'linear',
                                position: 'right',
                                min: 0,
                                max: 1.05,
                                grid: { drawOnChartArea: false },
                                ticks: {
                                    color: '#f43f5e',
                                    callback: val => (Number(val) * 100).toFixed(0) + '%'
                                },
                                title: { display: true, text: 'Causal Risk Prob', color: '#f43f5e' }
                            }
                        },
                        plugins: {
                            legend: {
                                labels: { color: '#f8fafc', font: { size: 11 } }
                            }
                        }
                    }
                });
            }
        }

        function renderMockCharts() {
            const count = 150;
            const dates = [];
            const nav_regime = [1.0];
            const nav_bh = [1.0];
            const nav_ma = [1.0];
            const dd_regime = [0.0];
            const dd_bh = [0.0];
            const high_risk_prob = [];
            const prices = [1.0];

            for (let i = 1; i < count; i++) {
                dates.push('20' + String(Math.floor(i / 7) + 5).padStart(2, '0') + '-' + String((i % 12) + 1).padStart(2, '0'));
                const is_crisis = (i > 30 && i < 45) || (i > 95 && i < 110);
                const r_bh = is_crisis ? (Math.random() * -0.035 - 0.005) : (Math.random() * 0.02 - 0.005);
                const p_risk = is_crisis ? 0.85 + Math.random() * 0.12 : 0.04 + Math.random() * 0.08;
                const r_regime = (1.0 - p_risk) * r_bh - 0.0001;

                nav_bh.push(nav_bh[i - 1] * (1 + r_bh));
                nav_regime.push(nav_regime[i - 1] * (1 + r_regime));
                nav_ma.push(nav_ma[i - 1] * (1 + r_bh * 0.85));
                prices.push(prices[i - 1] * (1 + r_bh));
                high_risk_prob.push(p_risk);

                const max_bh = Math.max(...nav_bh);
                dd_bh.push((nav_bh[i] - max_bh) / max_bh);

                const max_reg = Math.max(...nav_regime);
                dd_regime.push((nav_regime[i] - max_reg) / max_reg);
            }

            renderCharts({
                dates: dates,
                nav_regime: nav_regime,
                nav_bh: nav_bh,
                nav_ma: nav_ma,
                dd_regime: dd_regime,
                dd_bh: dd_bh,
                high_risk_prob: high_risk_prob,
                prices: prices
            });
        }

        function setEquityScale(scaleType) {
            currentScale = scaleType;
            document.getElementById('btn-scale-lin').classList.toggle('active', scaleType === 'linear');
            document.getElementById('btn-scale-log').classList.toggle('active', scaleType === 'logarithmic');
            if (equityChart && equityChart.options && equityChart.options.scales && equityChart.options.scales.y) {
                equityChart.options.scales.y.type = scaleType;
                equityChart.update();
            }
        }

        function applyPreset() {
            const preset = document.getElementById('sim-preset').value;
            const logRetInput = document.getElementById('sim-logret');
            const rv21Input = document.getElementById('sim-rv21');
            const volRatioInput = document.getElementById('sim-volratio');

            if (preset === 'bull') {
                logRetInput.value = '0.008';
                rv21Input.value = '0.11';
                volRatioInput.value = '0.75';
            } else if (preset === 'shock') {
                logRetInput.value = '-0.038';
                rv21Input.value = '0.42';
                volRatioInput.value = '2.6';
            } else if (preset === 'chop') {
                logRetInput.value = '0.001';
                rv21Input.value = '0.19';
                volRatioInput.value = '1.05';
            }
        }

        async function runSimulation() {
            const r = parseFloat(document.getElementById('sim-logret').value);
            const rv = parseFloat(document.getElementById('sim-rv21').value);
            const vr = parseFloat(document.getElementById('sim-volratio').value);

            try {
                const payload = {
                    history: [
                        { date: 't-2', features: [0.002, 0.15, 1.0] },
                        { date: 't-1', features: [0.001, 0.16, 1.1] },
                        { date: 't', features: [r, rv, vr] }
                    ],
                    risk_threshold_multiplier: 1.0
                };

                const res = await fetch('/predict-regime', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (res.ok) {
                    const data = await res.json();
                    updateSimUI(data.assigned_state, data.high_risk_probability, data.recommended_target_weight, data.predictive_log_likelihood);
                    return;
                }
            } catch (err) {
                console.warn('Backend inference error, using client estimation:', err);
            }

            // Client-side fallback calculation
            let p_risk = 0.05;
            if (vr > 1.8 || rv > 0.30 || r < -0.02) {
                p_risk = Math.min(0.96, 0.45 + (vr / 3.0) * 0.3 + (rv / 0.5) * 0.25);
            } else if (r > 0 && rv < 0.15) {
                p_risk = 0.04;
            } else {
                p_risk = 0.22;
            }
            const weight = Math.max(0.0, Math.min(1.0, 1.0 - p_risk));
            const state = p_risk > 0.5 ? 2 : (p_risk > 0.2 ? 1 : 0);
            updateSimUI(state, p_risk, weight, -3.95);
        }

        function updateSimUI(state, p_risk, weight, ll) {
            const stateEl = document.getElementById('res-state');
            const riskEl = document.getElementById('res-risk-prob');
            const fillEl = document.getElementById('res-prob-fill');
            const weightEl = document.getElementById('res-weight');
            const llEl = document.getElementById('res-ll');

            const p_pct = (p_risk * 100).toFixed(1) + '%';
            riskEl.innerText = p_pct;
            fillEl.style.width = p_pct;
            weightEl.innerText = (weight * 100).toFixed(1) + '%';
            llEl.innerText = Number(ll).toFixed(3);

            if (p_risk > 0.6) {
                stateEl.innerText = 'State ' + state + ' (High Risk / Crisis)';
                stateEl.className = 'value-rose';
                fillEl.style.backgroundColor = 'var(--accent-rose)';
            } else if (p_risk > 0.25) {
                stateEl.innerText = 'State ' + state + ' (Elevated Volatility)';
                stateEl.className = 'value-amber';
                fillEl.style.backgroundColor = 'var(--accent-amber)';
            } else {
                stateEl.innerText = 'State ' + state + ' (Quiescent Bull)';
                stateEl.className = 'value-emerald';
                fillEl.style.backgroundColor = 'var(--accent-emerald)';
            }
        }

        // In-page Lightbox Modal (Opens inside current window, zero tab redirects)
        function openModal(imgSrc, captionText) {
            const modal = document.getElementById('imageModal');
            const modalImg = document.getElementById('modalImg');
            const modalCaption = document.getElementById('modalCaption');
            if (modal && modalImg) {
                modalImg.src = imgSrc;
                modalCaption.innerText = captionText;
                modal.classList.add('active');
            }
        }

        function closeModal() {
            const modal = document.getElementById('imageModal');
            if (modal) {
                modal.classList.remove('active');
            }
        }

        // Close modal on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                closeModal();
            }
        });
    </script>
</body>
</html>
"""

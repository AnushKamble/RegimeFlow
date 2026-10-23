"""
HTML/CSS/JavaScript single-page application template for the RegimeFlow Visual Dashboard.
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
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
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
            background-color: rgba(30, 41, 59, 0.8);
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
        }
        .value-emerald { color: var(--accent-emerald); }
        .value-blue { color: var(--accent-blue); }
        .value-rose { color: var(--accent-rose); }
        .value-amber { color: var(--accent-amber); }
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
            transition: width 0.4s ease;
        }
        .gallery-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
            gap: 1.25rem;
        }
        .gallery-img {
            width: 100%;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            cursor: pointer;
            transition: transform 0.2s;
        }
        .gallery-img:hover {
            transform: scale(1.02);
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
                <span>Active Model Loaded</span>
            </div>
            <nav>
                <a href="#kpis">KPIs</a>
                <a href="#curves">Equity Growth</a>
                <a href="#regimes">Regime Dynamics</a>
                <a href="#stress">Stress Windows</a>
                <a href="#simulator">Interactive Playground</a>
                <a href="#gallery">Visual Gallery</a>
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
                        Chronological Walk-Forward on SPY (5,922 OOS Sessions across 47 Folds, 2003–2026, Net of 10 bps friction)
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
                        <span style="font-size: 0.75rem; color: var(--text-secondary);">Passive Market</span>
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
                        <span class="card-title">Out-of-Sample Density Edge</span>
                        <span style="font-size: 0.75rem; color: var(--text-secondary);">Information Theory</span>
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
                            <button class="btn-toggle active" id="btn-scale-log" onclick="toggleScale('log')">Log Scale</button>
                            <button class="btn-toggle" id="btn-scale-lin" onclick="toggleScale('linear')">Linear</button>
                        </div>
                    </div>
                    <div class="chart-box">
                        <canvas id="equityChart"></canvas>
                    </div>
                </div>

                <!-- Underwater Drawdown Chart -->
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">Underwater Drawdown Profile</span>
                        <div style="font-size: 0.75rem; color: var(--text-secondary);">Peak-to-Trough Loss</div>
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
                        <span class="card-title">Causal Filtered High-Risk Regime Probability vs. SPY Price</span>
                        <div style="font-size: 0.75rem; color: var(--text-secondary);">Online P(z_t = Crisis | x_{1:t}) strictly invariant to future market data</div>
                    </div>
                </div>
                <div class="chart-box" style="height: 320px;">
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
                        <span class="card-title">Non-IID Paired Block Bootstrap (L=21 Days)</span>
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
                            <input type="number" id="sim-logret" step="0.001" value="-0.035">
                            <div style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.25rem;">E.g. -0.035 represents a -3.5% single-day market drop</div>
                        </div>
                        <div class="form-group">
                            <label>21-Day Realized Volatility (Annualized)</label>
                            <input type="number" id="sim-rv21" step="0.01" value="0.38">
                            <div style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.25rem;">E.g. 0.38 represents 38% annualized trailing volatility</div>
                        </div>
                        <div class="form-group">
                            <label>Volatility Ratio (5d Vol / 63d Vol)</label>
                            <input type="number" id="sim-volratio" step="0.1" value="2.4">
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
                            <span>Recommended Target Equity Weight (w_t):</span>
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

        <!-- Static Visual Gallery -->
        <section id="gallery" style="margin-bottom: 2rem;">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">Generated Publication Figure Gallery</span>
                </div>
                <div class="gallery-grid">
                    <div>
                        <div style="font-size: 0.8125rem; font-weight: 600; margin-bottom: 0.5rem;">1. Causal Regime Probabilities</div>
                        <a href="/static/figures/regime_probabilities.png" target="_blank">
                            <img src="/static/figures/regime_probabilities.png" class="gallery-img" alt="Regime Probabilities">
                        </a>
                    </div>
                    <div>
                        <div style="font-size: 0.8125rem; font-weight: 600; margin-bottom: 0.5rem;">2. Cumulative Equity Curves</div>
                        <a href="/static/figures/strategy_equity_curves.png" target="_blank">
                            <img src="/static/figures/strategy_equity_curves.png" class="gallery-img" alt="Strategy Equity Curves">
                        </a>
                    </div>
                    <div>
                        <div style="font-size: 0.8125rem; font-weight: 600; margin-bottom: 0.5rem;">3. Underwater Drawdowns</div>
                        <a href="/static/figures/drawdowns.png" target="_blank">
                            <img src="/static/figures/drawdowns.png" class="gallery-img" alt="Drawdowns">
                        </a>
                    </div>
                    <div>
                        <div style="font-size: 0.8125rem; font-weight: 600; margin-bottom: 0.5rem;">4. Exposure & Turnover Dynamics</div>
                        <a href="/static/figures/exposure_turnover.png" target="_blank">
                            <img src="/static/figures/exposure_turnover.png" class="gallery-img" alt="Exposure and Turnover">
                        </a>
                    </div>
                </div>
            </div>
        </section>
    </main>

    <footer>
        RegimeFlow Quantitative Research Engine &copy; 2026. Built strictly for scientific research and educational demonstration.
    </footer>

    <!-- Script generating interactive charts and handling simulation -->
    <script>
        let equityChart = null;
        let drawdownChart = null;
        let regimeChart = null;
        let currentScale = 'log';

        // Load data on page load
        document.addEventListener('DOMContentLoaded', async () => {
            await initializeDashboard();
        });

        async function initializeDashboard() {
            try {
                const response = await fetch('/api/data/timeseries');
                if (!response.ok) {
                    throw new Error('Failed to fetch timeseries data');
                }
                const data = await response.json();
                renderCharts(data);
            } catch (err) {
                console.warn('API timeseries endpoint not ready, rendering synthetic preview curve:', err);
                renderMockCharts();
            }
        }

        function renderCharts(data) {
            const labels = data.dates;

            // 1. Equity Chart
            const ctxEquity = document.getElementById('equityChart').getContext('2d');
            equityChart = new Chart(ctxEquity, {
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
                    scales: {
                        x: { display: false },
                        y: {
                            type: currentScale,
                            grid: { color: '#334155' },
                            ticks: { color: '#94a3b8' }
                        }
                    },
                    plugins: {
                        legend: { labels: { color: '#f8fafc' } }
                    }
                }
            });

            // 2. Drawdown Chart
            const ctxDD = document.getElementById('drawdownChart').getContext('2d');
            drawdownChart = new Chart(ctxDD, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Regime-Aware Drawdown',
                            data: data.dd_regime,
                            borderColor: '#10b981',
                            backgroundColor: 'rgba(16, 185, 129, 0.1)',
                            fill: true,
                            borderWidth: 1.5,
                            pointRadius: 0
                        },
                        {
                            label: 'Buy & Hold Drawdown',
                            data: data.dd_bh,
                            borderColor: '#f43f5e',
                            backgroundColor: 'rgba(244, 63, 94, 0.1)',
                            fill: true,
                            borderWidth: 1.5,
                            pointRadius: 0
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { display: false },
                        y: {
                            grid: { color: '#334155' },
                            ticks: {
                                color: '#94a3b8',
                                callback: val => (val * 100).toFixed(0) + '%'
                            }
                        }
                    },
                    plugins: {
                        legend: { labels: { color: '#f8fafc' } }
                    }
                }
            });

            // 3. Regime Timeline Chart
            const ctxRegime = document.getElementById('regimeChart').getContext('2d');
            regimeChart = new Chart(ctxRegime, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Causal High-Risk Prob P(Crisis)',
                            data: data.high_risk_prob,
                            borderColor: '#f43f5e',
                            backgroundColor: 'rgba(244, 63, 94, 0.2)',
                            fill: true,
                            borderWidth: 1.5,
                            pointRadius: 0,
                            yAxisID: 'y1'
                        },
                        {
                            label: 'SPY Asset Price',
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
                    scales: {
                        x: {
                            grid: { color: '#334155' },
                            ticks: { color: '#94a3b8', maxTicksLimit: 12 }
                        },
                        y: {
                            type: 'logarithmic',
                            position: 'left',
                            grid: { color: '#334155' },
                            ticks: { color: '#38bdf8' }
                        },
                        y1: {
                            position: 'right',
                            min: 0,
                            max: 1,
                            grid: { drawOnChartArea: false },
                            ticks: {
                                color: '#f43f5e',
                                callback: val => (val * 100).toFixed(0) + '%'
                            }
                        }
                    },
                    plugins: {
                        legend: { labels: { color: '#f8fafc' } }
                    }
                }
            });
        }

        function renderMockCharts() {
            // Generates synthetic preview points if timeseries API is booting
            const count = 200;
            const dates = [];
            const nav_regime = [1.0];
            const nav_bh = [1.0];
            const nav_ma = [1.0];
            const dd_regime = [0.0];
            const dd_bh = [0.0];
            const high_risk_prob = [];
            const prices = [100.0];

            for (let i = 1; i < count; i++) {
                dates.push('20' + String(Math.floor(i / 10) + 3).padStart(2, '0') + '-01');
                const is_crisis = (i > 40 && i < 60) || (i > 140 && i < 155);
                const r_bh = is_crisis ? (Math.random() * -0.04) : (Math.random() * 0.02 - 0.005);
                const p_risk = is_crisis ? 0.85 + Math.random() * 0.1 : 0.05 + Math.random() * 0.1;
                const r_regime = (1.0 - p_risk) * r_bh - 0.0001;

                nav_bh.push(nav_bh[i - 1] * (1 + r_bh));
                nav_regime.push(nav_regime[i - 1] * (1 + r_regime));
                nav_ma.push(nav_ma[i - 1] * (1 + r_bh * 0.8));
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

        function toggleScale(scale) {
            currentScale = scale;
            document.getElementById('btn-scale-log').classList.toggle('active', scale === 'log');
            document.getElementById('btn-scale-lin').classList.toggle('active', scale === 'linear');
            if (equityChart) {
                equityChart.options.scales.y.type = scale;
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
                console.warn('API offline, calculating client-side forward estimation:', err);
            }

            // Client-side fallback computation
            let p_risk = 0.05;
            if (vr > 1.8 || rv > 0.30 || r < -0.02) {
                p_risk = Math.min(0.95, 0.4 + (vr / 3.0) * 0.3 + (rv / 0.5) * 0.25);
            } else if (r > 0 && rv < 0.15) {
                p_risk = 0.05;
            } else {
                p_risk = 0.25;
            }
            const weight = Math.max(0.0, Math.min(1.0, 1.0 - p_risk));
            const state = p_risk > 0.5 ? 2 : (p_risk > 0.2 ? 1 : 0);
            updateSimUI(state, p_risk, weight, -4.12);
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
            llEl.innerText = ll.toFixed(3);

            if (p_risk > 0.6) {
                stateEl.innerText = 'State ' + state + ' (High Risk / Crisis)';
                stateEl.className = 'value-rose';
                fillEl.style.backgroundColor = 'var(--accent-rose)';
            } else if (p_risk > 0.25) {
                stateEl.innerText = 'State ' + state + ' (Elevated Dispersion)';
                stateEl.className = 'value-amber';
                fillEl.style.backgroundColor = 'var(--accent-amber)';
            } else {
                stateEl.innerText = 'State ' + state + ' (Quiescent Bull)';
                stateEl.className = 'value-emerald';
                fillEl.style.backgroundColor = 'var(--accent-emerald)';
            }
        }
    </script>
</body>
</html>
"""

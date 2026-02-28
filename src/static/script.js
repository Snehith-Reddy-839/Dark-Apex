// Theme toggle stored in localStorage
function setTheme(theme) {
    // Set attribute on html element
    document.documentElement.setAttribute('data-theme', theme);
    document.documentElement.setAttribute('data-bs-theme', theme);
    localStorage.setItem('theme', theme);

    // Update button icon
    const toggle = document.getElementById('themeToggle');
    if (toggle) {
        if (theme === 'dark') {
            toggle.textContent = '🌙';
            toggle.title = 'Switch to Light Mode';
        } else {
            toggle.textContent = '☀️';
            toggle.title = 'Switch to Dark Mode';
        }
    }

    // Trigger repaint
    document.body.style.transition = 'background 0.4s ease, color 0.4s ease';

    // Update Plotly graphs after a small delay to ensure DOM is updated
    setTimeout(() => {
        updatePlotlyGraphs(theme);
    }, 100);
}

// Update Plotly chart colors to match current theme
function updatePlotlyGraphs(theme) {
    if (typeof Plotly === 'undefined') return;

    const isDark = theme === 'dark';
    const fontColor = isDark ? '#e0e0e0' : '#333333';
    const gridColor = isDark ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.1)';

    // Find all Plotly graph divs
    document.querySelectorAll('.plotly-graph-div').forEach(graphDiv => {
        if (graphDiv) {
            try {
                const layout = {
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: {
                        color: fontColor,
                        family: 'Arial, sans-serif',
                        size: 11
                    },
                    title: {
                        font: {
                            color: fontColor,
                            size: 16
                        }
                    },
                    xaxis: {
                        tickfont: { color: fontColor, size: 11 },
                        title_font: { color: fontColor, size: 12 },
                        gridcolor: gridColor,
                        zerolinecolor: fontColor
                    },
                    yaxis: {
                        tickfont: { color: fontColor, size: 11 },
                        title_font: { color: fontColor, size: 12 },
                        gridcolor: gridColor,
                        zerolinecolor: fontColor
                    },
                    legend: {
                        font: { color: fontColor, size: 11 }
                    }
                };

                Plotly.relayout(graphDiv, layout).catch(() => { });
            } catch (e) {
                // Silent fail
            }
        }
    });
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || 'light';
    const newTheme = current === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
}

function animateGraphDiv(graphDiv, forceReplay = false) {
    if (!graphDiv) return;
    if (!forceReplay && graphDiv.dataset.animated === 'true') return;
    if (!Array.isArray(graphDiv.data) || graphDiv.data.length === 0) return;

    const animatedTraces = [];
    const originalYByTrace = {};
    const resetPromises = [];
    const hideDuringReset = forceReplay;

    if (hideDuringReset) {
        graphDiv.style.opacity = '0';
    }

    graphDiv.data.forEach((trace, index) => {
        if (!trace || trace.type === 'pie' || !Array.isArray(trace.y) || trace.y.length === 0) {
            return;
        }

        const originalY = trace.y.map(value => Number(value) || 0);
        originalYByTrace[index] = originalY;
        animatedTraces.push(index);

        try {
            const resetPromise = Plotly.restyle(graphDiv, { y: [new Array(originalY.length).fill(0)] }, [index]);
            if (resetPromise && typeof resetPromise.then === 'function') {
                resetPromises.push(resetPromise);
            }
        } catch (e) {
            // Silent fail for non-animatable traces
        }
    });

    if (animatedTraces.length === 0) {
        const yAxis = graphDiv._fullLayout && graphDiv._fullLayout.yaxis;
        let yMin = 0;
        let yMax = 0;

        if (
            yAxis &&
            Array.isArray(yAxis.range) &&
            yAxis.range.length === 2 &&
            Number.isFinite(Number(yAxis.range[0])) &&
            Number.isFinite(Number(yAxis.range[1]))
        ) {
            const rangeStart = Number(yAxis.range[0]);
            const rangeEnd = Number(yAxis.range[1]);
            yMin = Math.min(rangeStart, rangeEnd, 0);
            yMax = Math.max(rangeStart, rangeEnd, 0);
        }

        if (yMax > 0) {
            try {
                Plotly.relayout(graphDiv, {
                    'yaxis.autorange': false,
                    'yaxis.range': [yMin, yMin]
                });

                setTimeout(() => {
                    if (hideDuringReset) {
                        graphDiv.style.opacity = '1';
                    }
                    try {
                        Plotly.animate(
                            graphDiv,
                            {
                                layout: {
                                    'yaxis.autorange': false,
                                    'yaxis.range': [yMin, yMax]
                                }
                            },
                            {
                                transition: { duration: 750, easing: 'cubic-in-out' },
                                frame: { duration: 750, redraw: false }
                            }
                        );
                    } catch (e) {
                        // Silent fail
                    }
                }, 80);
            } catch (e) {
                // Silent fail
            }
        }

        if (hideDuringReset && yMax <= 0) {
            graphDiv.style.opacity = '1';
        }

        graphDiv.dataset.animated = 'true';
        return;
    }

    Promise.allSettled(resetPromises).then(() => {
        if (hideDuringReset) {
            graphDiv.style.opacity = '1';
        }
        animatedTraces.forEach(index => {
            try {
                Plotly.animate(
                    graphDiv,
                    {
                        data: [{ y: originalYByTrace[index] }],
                        traces: [index]
                    },
                    {
                        transition: { duration: 750, easing: 'cubic-in-out' },
                        frame: { duration: 750, redraw: false }
                    }
                );
            } catch (e) {
                // Silent fail
            }
        });
        graphDiv.dataset.animated = 'true';
    });
}

function animatePlotlyGraphs() {
    if (typeof Plotly === 'undefined') return;

    document.querySelectorAll('.plotly-graph-div').forEach(graphDiv => {
        animateGraphDiv(graphDiv, false);
    });
}

window.addEventListener('DOMContentLoaded', () => {
    // Check for saved theme preference, otherwise use system preference
    const saved = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const initialTheme = saved || (prefersDark ? 'dark' : 'light');
    setTheme(initialTheme);

    setTimeout(() => {
        animatePlotlyGraphs();
    }, 250);

    document.querySelectorAll('.graph-replay-btn').forEach(button => {
        button.addEventListener('click', () => {
            const chartId = button.getAttribute('data-chart-id');
            const graphDiv = chartId ? document.getElementById(chartId) : null;
            if (!graphDiv || typeof Plotly === 'undefined') {
                return;
            }
            animateGraphDiv(graphDiv, true);
        });
    });

    // Attach theme toggle listener
    const toggle = document.getElementById('themeToggle');
    if (toggle) {
        toggle.addEventListener('click', toggleTheme);
    }

    // add spinner to AI form submit
    const aiForm = document.getElementById('aiForm');
    if (aiForm) {
        aiForm.addEventListener('submit', () => {
            const btn = aiForm.querySelector('button[type=submit]');
            if (btn) {
                btn.disabled = true;
                btn.innerHTML = 'Submitting <span class="spinner"></span>';
            }
        });
    }

    // Make graphs responsive
    window.addEventListener('resize', () => {
        document.querySelectorAll('.plotly-graph-div').forEach(graphDiv => {
            if (graphDiv) {
                try {
                    Plotly.Plots.resize(graphDiv);
                } catch (e) {
                    // Silent fail
                }
            }
        });
    });
});

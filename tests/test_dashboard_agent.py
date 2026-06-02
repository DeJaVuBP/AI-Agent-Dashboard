import pandas as pd

from ai_agent.dashboard_agent import DashboardAgent


def sample_df():
    return pd.DataFrame(
        {
            "Order Date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
            "Sales": [100.0, 200.0],
            "Profit": [10.0, 20.0],
            "Region": ["East", "West"],
            "Category": ["Furniture", "Office Supplies"],
            "Postal Code": [42420, 90036],
        }
    )


def test_infer_roles():
    df = sample_df()
    agent = DashboardAgent(df)
    summary = agent.dataset_summary()

    assert "Sales" in summary["metrics"] or "Profit" in summary["metrics"]
    assert "Region" in summary["dimensions"]
    assert "Order Date" in summary["dates"]


def test_request_parsing_bar_by():
    df = sample_df()
    agent = DashboardAgent(df)
    plan, response = agent.create_chart_plan_from_request("bar chart of Sales by Region")

    assert plan is not None
    assert plan.chart_type == "bar"
    assert plan.x == "Region"
    assert plan.y == "Sales"


def test_request_parsing_line_over():
    df = sample_df()
    agent = DashboardAgent(df)
    plan, response = agent.create_chart_plan_from_request("line chart of Profit over Order Date")

    assert plan is not None
    assert plan.chart_type == "line"
    assert plan.x == "Order Date"
    assert plan.y == "Profit"

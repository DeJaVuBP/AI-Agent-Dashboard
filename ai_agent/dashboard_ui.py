import pandas as pd
import plotly.express as px
import streamlit as st

from ai_agent.dashboard_agent import ChartPlan, DashboardAgent
from ai_agent.data_io import read_data


def build_chart(df: pd.DataFrame, plan: ChartPlan):
    if plan.chart_type == "line":
        return px.line(df.sort_values(plan.x), x=plan.x, y=plan.y, title=plan.title)
    if plan.chart_type == "bar":
        if plan.y in df.columns and pd.api.types.is_numeric_dtype(df[plan.y]) and plan.y != plan.x:
            grouped = df.groupby(plan.x, dropna=False)[plan.y].sum().reset_index()
            return px.bar(grouped, x=plan.x, y=plan.y, title=plan.title)
        grouped = df.groupby(plan.x, dropna=False).size().reset_index(name="count")
        return px.bar(grouped, x=plan.x, y="count", title=plan.title)
    if plan.chart_type == "scatter":
        return px.scatter(df, x=plan.x, y=plan.y, title=plan.title)
    if plan.chart_type == "area":
        return px.area(df.sort_values(plan.x), x=plan.x, y=plan.y, title=plan.title)
    if plan.chart_type == "pie":
        if plan.y in df.columns and pd.api.types.is_numeric_dtype(df[plan.y]) and plan.y != plan.x:
            grouped = df.groupby(plan.x, dropna=False)[plan.y].sum().reset_index()
            return px.pie(grouped, names=plan.x, values=plan.y, title=plan.title)
        grouped = df.groupby(plan.x, dropna=False).size().reset_index(name="count")
        return px.pie(grouped, names=plan.x, values="count", title=plan.title)
    return px.histogram(df, x=plan.x, title=plan.title)


def run_app():
    st.set_page_config(page_title="AI Excel Dashboard Agent", layout="wide")
    st.title("AI Agent: Excel to Dashboard")
    st.write("Upload an Excel or CSV file and let the agent create an automatic dashboard.")

    uploaded_file = st.file_uploader("Upload .xlsx, .xls, or .csv file", type=["xlsx", "xls", "csv"])
    if uploaded_file is None:
        st.info("Upload an Excel file to start.")
        return

    try:
        df, selected_sheet = read_data(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read Excel file: {exc}")
        return

    if df.empty:
        st.warning("The selected sheet is empty.")
        return

    agent = DashboardAgent(df)
    summary = agent.dataset_summary()
    exploration = agent.explore_data()

    if "chart_requests" not in st.session_state:
        st.session_state.chart_requests = []

    st.subheader("Dataset Overview")
    st.caption(f"Sheet: {selected_sheet}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", summary["rows"])
    c2.metric("Columns", summary["columns"])
    c3.metric("Missing Values", summary["missing_values"])
    c4.metric("Numeric Columns", len(summary["numeric"]))

    with st.expander("Column types"):
        st.write("Numeric:", summary["numeric"])
        st.write("Categorical:", summary["categorical"])
        st.write("Datetime:", summary["datetime"])
        st.write("Inferred metrics:", summary["metrics"])
        st.write("Inferred dimensions:", summary["dimensions"])
        st.write("Inferred dates:", summary["dates"])

    with st.expander("Data exploration"):
        if not exploration["numeric_profile"].empty:
            st.dataframe(exploration["numeric_profile"], use_container_width=True)
        if exploration["categorical_profile"]:
            st.write("Top categorical values")
            st.json(exploration["categorical_profile"])
        st.write("Missing values by column")
        st.json(exploration["missing_by_column"])

    st.subheader("Data Preview")
    st.dataframe(df.head(50), use_container_width=True)

    st.subheader("Create a Chart by Chat")

    for item in st.session_state.chart_requests:
        with st.chat_message("user"):
            st.write(item["request"])
        with st.chat_message("assistant"):
            st.write(item["response"])
            if item.get("plan"):
                fig = build_chart(agent.df, ChartPlan(**item["plan"]))
                st.plotly_chart(fig, use_container_width=True)

    request_text = st.chat_input("Ask for a chart using column names")
    if request_text:
        plan, response = agent.create_chart_plan_from_request(request_text)
        st.session_state.chart_requests.append(
            {
                "request": request_text,
                "response": response,
                "plan": plan.__dict__ if plan else None,
            }
        )
        st.rerun()

    st.subheader("Auto-Generated Dashboard")
    plans = agent.create_chart_plan()

    if not plans:
        st.warning("Not enough usable columns to build charts automatically.")
    else:
        for plan in plans[:4]:
            fig = build_chart(agent.df, plan)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("AI Insights (Optional)")
    st.write(
        "If GOOGLE_API_KEY is set in your environment or .env file, "
        "the agent will generate Gemini insights."
    )

    if st.button("Generate AI Insights"):
        with st.spinner("Generating insights with Gemini..."):
            insights = agent.generate_insights_with_gemini(summary)
        if insights:
            st.markdown(insights)
        else:
            st.info("GOOGLE_API_KEY not configured, or google-genai is unavailable.")

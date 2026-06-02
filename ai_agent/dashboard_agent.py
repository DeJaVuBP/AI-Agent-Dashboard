import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

try:
    import yaml
except Exception:
    yaml = None

try:
    from google import genai
except Exception:  # pragma: no cover - optional dependency at runtime
    genai = None


def _load_hints_from_config() -> Optional[Dict[str, List[str]]]:
    if yaml is None:
        return None
    root = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(root, "config", "column_hints.yaml")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
            return data
    except Exception:
        return None


@dataclass
class ChartPlan:
    title: str
    chart_type: str
    x: str
    y: str


class DashboardAgent:
    """Agent that reads tabular data and proposes a dashboard automatically."""

    METRIC_NAME_HINTS = (
        "sales",
        "profit",
        "revenue",
        "amount",
        "cost",
        "price",
        "quantity",
        "discount",
        "margin",
        "units",
        "score",
        "value",
    )
    DIMENSION_NAME_HINTS = (
        "category",
        "segment",
        "region",
        "state",
        "city",
        "country",
        "sub category",
        "ship mode",
        "customer",
        "product",
        "market",
        "group",
    )
    DATE_NAME_HINTS = (
        "date",
        "time",
        "month",
        "quarter",
        "year",
        "order date",
        "ship date",
    )
    ID_NAME_HINTS = (
        "id",
        "code",
        "postal",
        "zip",
        "phone",
        "fax",
        "account",
        "number",
    )

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()
        # allow config-driven hints
        cfg = _load_hints_from_config()
        if cfg:
            self.metric_hints = tuple(cfg.get("metrics", self.METRIC_NAME_HINTS))
            self.dimension_hints = tuple(cfg.get("dimensions", self.DIMENSION_NAME_HINTS))
            self.date_hints = tuple(cfg.get("dates", self.DATE_NAME_HINTS))
            self.id_hints = tuple(cfg.get("ids", self.ID_NAME_HINTS))
        else:
            self.metric_hints = self.METRIC_NAME_HINTS
            self.dimension_hints = self.DIMENSION_NAME_HINTS
            self.date_hints = self.DATE_NAME_HINTS
            self.id_hints = self.ID_NAME_HINTS

    def _numeric_columns(self) -> List[str]:
        return self.df.select_dtypes(include=["number"]).columns.tolist()

    def _categorical_columns(self) -> List[str]:
        return self.df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    def _datetime_columns(self) -> List[str]:
        datetime_cols = self.df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()

        for col in self.df.select_dtypes(include=["object"]).columns:
            parsed = pd.to_datetime(self.df[col], errors="coerce")
            if parsed.notna().mean() > 0.8:
                self.df[col] = parsed
                datetime_cols.append(col)

        return list(dict.fromkeys(datetime_cols))

    def _normalized_name(self, column: str) -> str:
        return column.replace("_", " ").replace("-", " ").lower()

    def _name_matches(self, column: str, hints: Tuple[str, ...]) -> bool:
        normalized = self._normalized_name(column)
        return any(hint in normalized for hint in hints)

    def _matching_columns(self, text: str) -> List[str]:
        normalized_text = self._normalized_name(text)
        matches = []
        for column in sorted(self.df.columns, key=lambda item: len(self._normalized_name(item)), reverse=True):
            if self._normalized_name(column) in normalized_text:
                matches.append(column)
        return matches

    def _first_matching_column(self, text: str, candidates: List[str]) -> Optional[str]:
        normalized_text = self._normalized_name(text)
        for column in candidates:
            if self._normalized_name(column) in normalized_text:
                return column
        return None

    def _detect_chart_type(self, request_text: str) -> str:
        if any(word in request_text for word in ("line chart", "trend", "over time", "time series")):
            return "line"
        if any(word in request_text for word in ("scatter", "relationship", "correlation", "vs", "against", "compare")):
            return "scatter"
        if any(word in request_text for word in ("pie", "donut")):
            return "pie"
        if any(word in request_text for word in ("histogram", "distribution", "frequency")):
            return "histogram"
        if any(word in request_text for word in ("area",)):
            return "area"
        if any(word in request_text for word in ("bar chart", "bar", "column chart", "by ")):
            return "bar"
        return ""

    def _infer_column_roles(self) -> Dict[str, List[str]]:
        numeric_cols = self._numeric_columns()
        categorical_cols = self._categorical_columns()
        datetime_cols = self._datetime_columns()

        metrics = [
            column
            for column in numeric_cols
            if self._name_matches(column, self.metric_hints)
            and not self._name_matches(column, self.id_hints)
        ]
        if not metrics:
            metrics = [
                column
                for column in numeric_cols
                if not self._name_matches(column, self.id_hints)
            ]

        dimensions = [
            column
            for column in categorical_cols
            if self._name_matches(column, self.dimension_hints)
            and not self._name_matches(column, self.date_hints)
        ]
        if not dimensions:
            dimensions = [
                column
                for column in categorical_cols
                if column not in datetime_cols and not self._name_matches(column, self.id_hints)
            ]

        dates = [
            column
            for column in datetime_cols
            if self._name_matches(column, self.date_hints) or column not in metrics
        ]

        return {
            "metrics": list(dict.fromkeys(metrics)),
            "dimensions": list(dict.fromkeys(dimensions)),
            "dates": list(dict.fromkeys(dates)),
        }

    def dataset_summary(self) -> Dict[str, object]:
        roles = self._infer_column_roles()
        return {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "numeric": self._numeric_columns(),
            "categorical": self._categorical_columns(),
            "datetime": self._datetime_columns(),
            "metrics": roles["metrics"],
            "dimensions": roles["dimensions"],
            "dates": roles["dates"],
            "missing_values": int(self.df.isna().sum().sum()),
        }

    def explore_data(self) -> Dict[str, object]:
        roles = self._infer_column_roles()
        numeric_cols = self._numeric_columns()
        categorical_cols = self._categorical_columns()
        datetime_cols = self._datetime_columns()

        numeric_profile = (
            self.df[numeric_cols].describe().T.reset_index().rename(columns={"index": "column"})
            if numeric_cols
            else pd.DataFrame()
        )

        categorical_profile = []
        for column in categorical_cols[:5]:
            top_values = self.df[column].value_counts(dropna=False).head(5)
            categorical_profile.append(
                {
                    "column": column,
                    "top_values": top_values.to_dict(),
                }
            )

        return {
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "datetime_columns": datetime_cols,
            "metrics": roles["metrics"],
            "dimensions": roles["dimensions"],
            "dates": roles["dates"],
            "missing_by_column": self.df.isna().sum().sort_values(ascending=False).to_dict(),
            "numeric_profile": numeric_profile,
            "categorical_profile": categorical_profile,
        }

    def create_chart_plan(self) -> List[ChartPlan]:
        exploration = self.explore_data()
        metric_cols = exploration["metrics"]
        dimension_cols = exploration["dimensions"]
        date_cols = exploration["dates"]

        plans: List[ChartPlan] = []

        if date_cols and metric_cols:
            plans.append(
                ChartPlan(
                    title=f"Trend of {metric_cols[0]} over {date_cols[0]}",
                    chart_type="line",
                    x=date_cols[0],
                    y=metric_cols[0],
                )
            )

        if dimension_cols and metric_cols:
            plans.append(
                ChartPlan(
                    title=f"{metric_cols[0]} by {dimension_cols[0]}",
                    chart_type="bar",
                    x=dimension_cols[0],
                    y=metric_cols[0],
                )
            )

        if len(metric_cols) >= 2:
            plans.append(
                ChartPlan(
                    title=f"{metric_cols[0]} vs {metric_cols[1]}",
                    chart_type="scatter",
                    x=metric_cols[0],
                    y=metric_cols[1],
                )
            )

        if metric_cols:
            plans.append(
                ChartPlan(
                    title=f"Distribution of {metric_cols[0]}",
                    chart_type="histogram",
                    x=metric_cols[0],
                    y=metric_cols[0],
                )
            )

        return plans

    def create_chart_plan_from_request(self, request: str) -> Tuple[Optional[ChartPlan], str]:
        request_text = request.strip()
        if not request_text:
            return None, "Type a chart request like 'bar chart of Sales by Region'."

        normalized_request = self._normalized_name(request_text)
        roles = self._infer_column_roles()
        metrics = roles["metrics"]
        dimensions = roles["dimensions"]
        dates = roles["dates"]
        matched_columns = self._matching_columns(request_text)

        chart_type = self._detect_chart_type(normalized_request)

        before_by, after_by = request_text, ""
        if " by " in normalized_request:
            before_by, after_by = request_text.rsplit(" by ", 1)

        before_over, after_over = request_text, ""
        if " over " in normalized_request:
            before_over, after_over = request_text.rsplit(" over ", 1)

        x_column: Optional[str] = None
        y_column: Optional[str] = None

        if chart_type in {"line", "area"} or " over " in normalized_request:
            x_column = self._first_matching_column(after_over, dates) or self._first_matching_column(
                after_over, matched_columns
            )
            y_column = self._first_matching_column(before_over, metrics) or self._first_matching_column(
                before_over, matched_columns
            )
            chart_type = chart_type or "line"
        elif chart_type in {"scatter"} or any(word in normalized_request for word in (" vs ", " versus ", " against ")):
            if len(metrics) >= 2:
                x_column, y_column = metrics[0], metrics[1]
            elif len(matched_columns) >= 2:
                x_column, y_column = matched_columns[0], matched_columns[1]
            chart_type = "scatter"
        elif chart_type in {"pie", "bar"} or " by " in normalized_request:
            x_column = self._first_matching_column(after_by, dimensions) or self._first_matching_column(
                after_by, matched_columns
            )
            y_column = self._first_matching_column(before_by, metrics) or self._first_matching_column(
                before_by, matched_columns
            )
            chart_type = "bar" if chart_type != "pie" else "pie"
        elif chart_type == "histogram":
            x_column = self._first_matching_column(request_text, metrics) or self._first_matching_column(
                request_text, matched_columns
            )
            chart_type = "histogram"
        else:
            if len(matched_columns) >= 2:
                x_column, y_column = matched_columns[0], matched_columns[1]
                chart_type = "scatter" if x_column in metrics and y_column in metrics else "bar"
            elif len(matched_columns) == 1:
                x_column = matched_columns[0]
                chart_type = "histogram" if x_column in metrics else "bar"

        if chart_type in {"line", "area", "scatter", "bar", "pie"} and not x_column:
            x_column = dimensions[0] if dimensions else dates[0] if dates else (matched_columns[0] if matched_columns else None)
        if chart_type in {"line", "area", "scatter", "bar", "pie"} and not y_column:
            y_column = metrics[0] if metrics else (matched_columns[0] if matched_columns else None)

        if chart_type == "histogram" and not x_column:
            x_column = metrics[0] if metrics else (matched_columns[0] if matched_columns else None)

        if not x_column:
            return None, (
                "I could not find the column names in your request. "
                f"Available columns: {', '.join(self.df.columns)}"
            )

        title_parts = [chart_type.replace("histogram", "distribution").title(), x_column]
        if y_column and chart_type not in {"histogram", "pie"}:
            title_parts.append(f"vs {y_column}" if chart_type == "scatter" else f"by {y_column}")

        return (
            ChartPlan(
                title=" ".join(title_parts),
                chart_type=chart_type,
                x=x_column,
                y=y_column or x_column,
            ),
            f"Created a {chart_type} chart using {x_column}" + (f" and {y_column}" if y_column else ""),
        )

    def generate_insights_with_gemini(self, summary: Dict[str, object]) -> Optional[str]:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or genai is None:
            return None

        try:
            client = genai.Client(api_key=api_key)
            prompt = (
                "You are a data analyst agent. Based on this dataset summary, "
                "write 5 concise, practical insights and 3 dashboard recommendations.\n\n"
                f"Summary: {summary}"
            )
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            return response.text
        except Exception as exc:
            return f"Gemini insight generation failed: {exc}"
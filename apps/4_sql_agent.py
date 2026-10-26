
# ============================================================
# 1. IMPORTS AND CONFIGURATION
# ============================================================
import os
from pathlib import Path
from dotenv import load_dotenv

import sqlite3
import uuid

import pandas as pd
import plotly.express as px
import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent


# ============================================================
# 2. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

DB_PATH = BASE_DIR / "my_tasks.db"
DB_URI = f"sqlite:///{DB_PATH.as_posix()}"

st.set_page_config(
    page_title="AI Business Intelligence Hub",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# 3. DATABASE INITIALIZATION
# ============================================================

db = SQLDatabase.from_uri(DB_URI)

# Keep the existing tasks table from your original project.
# The business dashboard uses browser_data.
db.run("""
    CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT,
        status TEXT CHECK (
            status IN ('pending', 'in progress', 'completed')
        ) NOT NULL DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")


# ============================================================
# 4. LOAD SALES DATA FROM SQLITE
# ============================================================

def quote_identifier(name: str) -> str:
    """Safely quote a SQLite identifier."""
    return '"' + name.replace('"', '""') + '"'


@st.cache_data(ttl=300)
def load_sales_data(
    db_path: str,
    db_modified_time: float
) -> pd.DataFrame:

    required_columns = [
        "Segment",
        "Country",
        "Product",
        "Discount Band",
        "Units Sold",
        "Manufacturing Price",
        "Sale Price",
        "Gross Sales",
        "Discounts",
        "Sales",
        "COGS",
        "Profit",
        "Date",
        "Month Number",
        "Month Name",
        "Year"
    ]

    with sqlite3.connect(db_path) as connection:

        # Find tables in the database
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type = 'table'",
            connection
        )["name"].tolist()

        if "browser_data" not in tables:
            raise ValueError(
                "The browser_data table was not found. "
                "Run load_excel.py first."
            )

        # Read the original column names
        schema = pd.read_sql_query(
            'PRAGMA table_info("browser_data")',
            connection
        )

        original_columns = schema["name"].tolist()

        # Map cleaned names to the original database column names.
        # This also handles an older column named ' Sales'.
        column_map = {}

        for original_name in original_columns:

            cleaned_name = str(original_name).strip()

            if cleaned_name in column_map:
                raise ValueError(
                    "Duplicate database columns after trimming "
                    f"spaces: {cleaned_name}"
                )

            column_map[cleaned_name] = original_name

        missing_columns = [
            column
            for column in required_columns
            if column not in column_map
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns in browser_data: {missing_columns}. "
                "Check your Excel file and rerun load_excel.py."
            )

        # Select required columns and return consistent clean names
        select_parts = []

        for column in required_columns:

            original_name = column_map[column]

            select_parts.append(
                f"{quote_identifier(original_name)} "
                f"AS {quote_identifier(column)}"
            )

        query = (
            'SELECT '
            + ", ".join(select_parts)
            + ' FROM "browser_data"'
        )

        df = pd.read_sql_query(query, connection)

    # --------------------------------------------------------
    # Convert metrics and date fields into usable data types
    # --------------------------------------------------------

    numeric_columns = [
        "Units Sold",
        "Manufacturing Price",
        "Sale Price",
        "Gross Sales",
        "Discounts",
        "Sales",
        "COGS",
        "Profit",
        "Month Number",
        "Year"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column]
            .astype(str)
            .str.replace(r"[$,\s]", "", regex=True),
            errors="coerce"
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # Clean fields used in the interactive filters
    category_columns = [
        "Segment",
        "Country",
        "Product",
        "Discount Band",
        "Month Name"
    ]

    for column in category_columns:

        df[column] = (
            df[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )

    return df


# ============================================================
# 5. INTERACTIVE SALES DASHBOARD
# ============================================================

def show_sales_dashboard(data: pd.DataFrame):

    st.title("📊 Sales Performance Dashboard")

    st.caption(
        "Explore revenue, profit, and product performance "
        "using interactive business intelligence."
    )

    # --------------------------------------------------------
    # Dashboard filters
    # --------------------------------------------------------

    year_options = sorted(
        data["Year"].dropna().astype(int).unique().tolist()
    )

    country_options = sorted(data["Country"].unique().tolist())
    product_options = sorted(data["Product"].unique().tolist())
    segment_options = sorted(data["Segment"].unique().tolist())

    discount_options = sorted(
        data["Discount Band"].unique().tolist()
    )

    with st.sidebar:

        st.header("Dashboard Filters")

        st.caption(
            "Select values to filter the dashboard. "
            "Clear a selection to include all values."
        )

        selected_years = st.multiselect(
            "Year",
            options=year_options,
            default=year_options,
            key="dashboard_year"
        )

        selected_countries = st.multiselect(
            "Country",
            options=country_options,
            default=country_options,
            key="dashboard_country"
        )

        selected_products = st.multiselect(
            "Product",
            options=product_options,
            default=product_options,
            key="dashboard_product"
        )

        selected_segments = st.multiselect(
            "Segment",
            options=segment_options,
            default=segment_options,
            key="dashboard_segment"
        )

        selected_discounts = st.multiselect(
            "Discount Band",
            options=discount_options,
            default=discount_options,
            key="dashboard_discount"
        )

    # --------------------------------------------------------
    # Apply filters
    # --------------------------------------------------------

    filtered = data.copy()

    # An empty selection means that all values are included.
    if selected_years:
        filtered = filtered[
            filtered["Year"].isin(selected_years)
        ]

    if selected_countries:
        filtered = filtered[
            filtered["Country"].isin(selected_countries)
        ]

    if selected_products:
        filtered = filtered[
            filtered["Product"].isin(selected_products)
        ]

    if selected_segments:
        filtered = filtered[
            filtered["Segment"].isin(selected_segments)
        ]

    if selected_discounts:
        filtered = filtered[
            filtered["Discount Band"].isin(selected_discounts)
        ]

    st.caption(
        f"Records matching selected filters: {len(filtered):,}"
    )

    if filtered.empty:
        st.warning(
            "No records match your selected filters. "
            "Clear a filter or choose other values."
        )
        return

    # --------------------------------------------------------
    # Calculate KPIs
    # --------------------------------------------------------

    total_sales = filtered["Sales"].sum()
    total_profit = filtered["Profit"].sum()
    total_units = filtered["Units Sold"].sum()

    # Weighted profit margin based on total sales
    profit_margin = (
        (total_profit / total_sales) * 100
        if total_sales != 0
        else 0.0
    )

    average_discount = filtered["Discounts"].mean()

    if pd.isna(average_discount):
        average_discount = 0.0

    # --------------------------------------------------------
    # Display KPI cards
    # --------------------------------------------------------

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        label="Total Sales",
        value=f"${total_sales:,.2f}"
    )

    col2.metric(
        label="Total Profit",
        value=f"${total_profit:,.2f}"
    )

    col3.metric(
        label="Profit Margin",
        value=f"{profit_margin:.2f}%"
    )

    col4.metric(
        label="Units Sold",
        value=f"{total_units:,.1f}"
    )

    col5.metric(
        label="Average Discount Amount",
        value=f"${average_discount:,.2f}"
    )

    st.divider()

    # --------------------------------------------------------
    # Monthly sales and profit analysis
    # --------------------------------------------------------

    monthly_source = filtered.dropna(
        subset=["Year", "Month Number"]
    ).copy()

    monthly = (
        monthly_source
        .groupby(
            ["Year", "Month Number", "Month Name"],
            as_index=False
        )
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum")
        )
        .sort_values(["Year", "Month Number"])
    )

    if monthly.empty:

        st.info(
            "Monthly charts are unavailable because "
            "Year or Month Number has no valid data."
        )

    else:

        # Preserve chronological order across multiple years
        monthly["Month"] = (
            monthly["Year"].astype(int).astype(str)
            + "-"
            + monthly["Month Number"]
                .astype(int)
                .astype(str)
                .str.zfill(2)
        )

        left, right = st.columns(2)

        # ---------------- Sales trend ----------------

        with left:

            sales_fig = px.line(
                monthly,
                x="Month",
                y="Sales",
                markers=True,
                hover_data=["Year", "Month Name"],
                title="Monthly Sales Trend"
            )

            sales_fig.update_yaxes(
                title="Sales ($)",
                tickprefix="$",
                tickformat=",.0f"
            )

            sales_fig.update_layout(
                xaxis_title="Month",
                hovermode="x unified"
            )

            st.plotly_chart(
                sales_fig,
                use_container_width=True
            )

        # ---------------- Profit trend ----------------

        with right:

            profit_fig = px.line(
                monthly,
                x="Month",
                y="Profit",
                markers=True,
                hover_data=["Year", "Month Name"],
                title="Monthly Profit Trend"
            )

            profit_fig.update_yaxes(
                title="Profit ($)",
                tickprefix="$",
                tickformat=",.0f"
            )

            profit_fig.update_layout(
                xaxis_title="Month",
                hovermode="x unified"
            )

            st.plotly_chart(
                profit_fig,
                use_container_width=True
            )

    # --------------------------------------------------------
    # Filtered records
    # --------------------------------------------------------

    with st.expander("View filtered sales data"):

        st.dataframe(
            filtered,
            use_container_width=True
        )

    # --------------------------------------------------------
    # Download filtered data
    # --------------------------------------------------------

    csv_data = filtered.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download filtered data as CSV",
        data=csv_data,
        file_name="filtered_sales_data.csv",
        mime="text/csv"
    )


# ============================================================
# 6. GROQ MODEL AND SQL AGENT
# ============================================================

try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error(
        "Groq API key is missing. Add GROQ_API_KEY "
        "in Streamlit Cloud Settings > Secrets."
    )
    st.stop()

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=groq_api_key
)

toolkit = SQLDatabaseToolkit(
    db=db,
    llm=model
)

tools = toolkit.get_tools()

system_prompt = """
You are an expert AI Business Intelligence Analyst
and SQL Assistant for corporate reporting.

DATABASE:
The primary business table is browser_data.

CATEGORICAL FIELDS:
"Segment", "Country", "Product",
"Discount Band", "Month Name"

METRIC FIELDS:
"Units Sold", "Manufacturing Price", "Sale Price",
"Gross Sales", "Discounts", "Sales", "COGS", "Profit"

TIME FIELDS:
"Date", "Month Number", "Year"

SQL GUIDELINES:
1. Use SQLite-compatible SQL.
2. Enclose column names containing spaces in double quotes.
3. Never use SELECT *.
4. Retrieve only columns required to answer the question.
5. Use SUM, AVG, MIN, MAX and COUNT when appropriate.
6. Use GROUP BY for category-level comparisons.
7. Use ORDER BY and LIMIT for ranking questions.
8. Limit detailed result sets to 10 rows by default.
9. Only query information from the database.
10. Do not create, alter, insert, update or delete records.
11. Inspect the schema if column names are uncertain.
12. Never invent numerical results or unsupported trends.

RESPONSE FORMAT:
1. Begin with a direct answer.
2. Present important figures clearly.
3. Provide two or three useful analytical observations.
4. Base all findings on executed query results.
"""


@st.cache_resource
def get_agent():

    return create_agent(
        model=model,
        tools=tools,
        checkpointer=InMemorySaver(),
        system_prompt=system_prompt
    )


agent = get_agent()


# ============================================================
# 7. APPLICATION TABS
# ============================================================

dashboard_tab, chat_tab = st.tabs([
    "📊 Sales Dashboard",
    "💬 AI SQL Assistant"
])


# ============================================================
# 8. SALES DASHBOARD TAB
# ============================================================

with dashboard_tab:

    try:

        if not DB_PATH.exists():
            raise FileNotFoundError(
                f"Database not found: {DB_PATH}. "
                "Run load_excel.py first."
            )

        sales_df = load_sales_data(
            str(DB_PATH),
            DB_PATH.stat().st_mtime
        )

        show_sales_dashboard(sales_df)

    except Exception as exc:

        st.error("Unable to load the sales dashboard.")

        st.write(
            "Verify data.xlsx, browser_data, the database path, "
            "and the required column names."
        )

        with st.expander("View error details"):
            st.exception(exc)


# ============================================================
# 9. AI SQL ASSISTANT TAB
# ============================================================

with chat_tab:

    st.subheader(
        "📃 Data Analytics Hub | Text-to-SQL Business Intelligence"
    )

    # A unique thread for each browser session
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous conversation
    for message in st.session_state.messages:

        role = (
            "assistant"
            if message["role"] in ("ai", "assistant")
            else "user"
        )

        st.chat_message(role).markdown(message["content"])

    prompt = st.chat_input(
        "Ask a business question about your sales data..."
    )

    if prompt:

        # Display and store the new question
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        st.chat_message("user").markdown(prompt)

        with st.chat_message("assistant"):

            with st.spinner("Analyzing your business question..."):

                try:

                    response = agent.invoke(
                        {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ]
                        },
                        {
                            "configurable": {
                                "thread_id": st.session_state.thread_id
                            }
                        }
                    )

                    result = response["messages"][-1].content

                    # Handle text-block responses if returned
                    if isinstance(result, list):

                        result = "\n".join(
                            block.get("text", "")
                            for block in result
                            if isinstance(block, dict)
                        )

                    st.markdown(result)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result
                    })

                except Exception:

                    st.error(
                        "I couldn't answer that question. "
                        "Check your database, API key, and model "
                        "configuration, then try again."
                    )

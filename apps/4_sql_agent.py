# ============================================================
# AI BUSINESS INTELLIGENCE HUB
# Sales Dashboard + AI SQL Assistant
# ============================================================

import os
import sqlite3
import uuid
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Business Intelligence Hub",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

DB_PATH = BASE_DIR / "my_tasks.db"
EXCEL_PATH = BASE_DIR / "data.xlsx"

REQUIRED_COLUMNS = [
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
    "Year",
]


# ============================================================
# 2. DATABASE INITIALIZATION
# ============================================================

def ensure_browser_data():
    """
    Create browser_data from data.xlsx if the table is missing
    or its required columns are incomplete.
    """

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Create the existing tasks table if needed.
    with sqlite3.connect(DB_PATH) as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                status TEXT CHECK (
                    status IN (
                        'pending',
                        'in progress',
                        'completed'
                    )
                ) NOT NULL DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        table_exists = connection.execute("""
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table'
              AND name = 'browser_data'
        """).fetchone()

        if table_exists:
            schema = pd.read_sql_query(
                'PRAGMA table_info("browser_data")',
                connection
            )

            existing_columns = {
                str(column).strip()
                for column in schema["name"].tolist()
            }
        else:
            existing_columns = set()

    # Reuse the table if all required columns already exist.
    if set(REQUIRED_COLUMNS).issubset(existing_columns):
        print("Existing browser_data table is valid.")
        return

    # A fresh cloud deployment must have the Excel file.
    if not EXCEL_PATH.is_file():
        raise FileNotFoundError(
            f"Could not find data.xlsx at: {EXCEL_PATH}. "
            "Ensure data.xlsx is committed to the repository root."
        )

    print(f"Loading sales data from: {EXCEL_PATH}")

    sales_df = pd.read_excel(EXCEL_PATH)

    # Fix headers like " Sales" by removing outer spaces.
    sales_df.columns = (
        sales_df.columns.astype(str).str.strip()
    )

    duplicates = sales_df.columns[
        sales_df.columns.duplicated()
    ].tolist()

    if duplicates:
        raise ValueError(
            f"Duplicate column names after cleanup: {duplicates}"
        )

    # Validate the Excel schema.
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in sales_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in data.xlsx: {missing_columns}. "
            f"Found columns: {sales_df.columns.tolist()}"
        )

    sales_df = sales_df[REQUIRED_COLUMNS].copy()

    # Store dates consistently in SQLite.
    sales_df["Date"] = pd.to_datetime(
        sales_df["Date"],
        errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    # Write the corrected sales table.
    # Existing malformed browser_data is replaced only when
    # its required schema is incomplete.
    with sqlite3.connect(DB_PATH) as connection:

        sales_df.to_sql(
            name="browser_data",
            con=connection,
            if_exists="replace",
            index=False
        )

        row_count = connection.execute(
            'SELECT COUNT(*) FROM "browser_data"'
        ).fetchone()[0]

    print(
        f"Database initialized successfully: {row_count} sales rows."
    )


# Initialize before creating the LangChain database connection.
try:
    ensure_browser_data()

except Exception as exc:

    st.error("Unable to initialize the sales database.")

    st.write(
        "Check that data.xlsx exists in the project root "
        "and contains the required sales columns."
    )

    with st.expander("View database initialization error"):
        st.exception(exc)

    st.stop()


# ============================================================
# 3. CONNECT LANGCHAIN TO SQLITE
# ============================================================

DB_URI = f"sqlite:///{DB_PATH.as_posix()}"

db = SQLDatabase.from_uri(
    DB_URI,
    include_tables=["browser_data"],
)


def quote_identifier(name: str) -> str:
    """Safely quote a SQLite identifier."""
    return '"' + name.replace('"', '""') + '"'


# ============================================================
# 4. LOAD AND CLEAN SALES DATA
# ============================================================

@st.cache_data(ttl=300)
def load_sales_data(
    db_path: str,
    db_modified_time: float
) -> pd.DataFrame:

    with sqlite3.connect(db_path) as connection:

        tables = pd.read_sql_query(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            """,
            connection
        )["name"].tolist()

        if "browser_data" not in tables:
            raise ValueError(
                "browser_data is missing after database initialization."
            )

        schema = pd.read_sql_query(
            'PRAGMA table_info("browser_data")',
            connection
        )

        original_columns = schema["name"].tolist()
        column_map = {}

        # Support old database headers such as " Sales".
        for original_name in original_columns:

            clean_name = str(original_name).strip()

            if clean_name in column_map:
                raise ValueError(
                    "Duplicate column names after trimming spaces: "
                    f"{clean_name}"
                )

            column_map[clean_name] = original_name

        missing_columns = [
            column
            for column in REQUIRED_COLUMNS
            if column not in column_map
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns in browser_data: {missing_columns}"
            )

        select_parts = [
            f"{quote_identifier(column_map[column])} "
            f"AS {quote_identifier(column)}"
            for column in REQUIRED_COLUMNS
        ]

        query = (
            "SELECT "
            + ", ".join(select_parts)
            + ' FROM "browser_data"'
        )

        df = pd.read_sql_query(query, connection)

    # Convert financial and other numeric fields.
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
        "Year",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column]
            .astype(str)
            .str.replace(r"[$,\s]", "", regex=True),
            errors="coerce",
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # Clean columns used in the sidebar filters.
    category_columns = [
        "Segment",
        "Country",
        "Product",
        "Discount Band",
        "Month Name",
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
        "Explore sales, profitability, and product performance."
    )

    # ---------------- Filter options ----------------

    year_options = sorted(
        data["Year"].dropna().astype(int).unique().tolist()
    )

    country_options = sorted(data["Country"].unique().tolist())
    product_options = sorted(data["Product"].unique().tolist())
    segment_options = sorted(data["Segment"].unique().tolist())

    discount_options = sorted(
        data["Discount Band"].unique().tolist()
    )

    # ---------------- Sidebar filters ----------------

    with st.sidebar:

        st.header("Dashboard Filters")

        st.caption(
            "Clear a selection to include all available values."
        )

        selected_years = st.multiselect(
            "Year",
            options=year_options,
            default=year_options,
            key="dashboard_year",
        )

        selected_countries = st.multiselect(
            "Country",
            options=country_options,
            default=country_options,
            key="dashboard_country",
        )

        selected_products = st.multiselect(
            "Product",
            options=product_options,
            default=product_options,
            key="dashboard_product",
        )

        selected_segments = st.multiselect(
            "Segment",
            options=segment_options,
            default=segment_options,
            key="dashboard_segment",
        )

        selected_discounts = st.multiselect(
            "Discount Band",
            options=discount_options,
            default=discount_options,
            key="dashboard_discount",
        )

    # ---------------- Apply filters ----------------

    filtered = data.copy()

    # An empty selection includes all values.
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
            "No records match your filters. "
            "Choose different values or clear a filter."
        )
        return

    # ---------------- Calculate KPIs ----------------

    total_sales = filtered["Sales"].sum()
    total_profit = filtered["Profit"].sum()
    total_units = filtered["Units Sold"].sum()

    profit_margin = (
        (total_profit / total_sales) * 100
        if total_sales != 0
        else 0.0
    )

    average_discount = filtered["Discounts"].mean()

    if pd.isna(average_discount):
        average_discount = 0.0

    # ---------------- Display KPI cards ----------------

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Sales", f"${total_sales:,.2f}")
    col2.metric("Total Profit", f"${total_profit:,.2f}")
    col3.metric("Profit Margin", f"{profit_margin:.2f}%")
    col4.metric("Units Sold", f"{total_units:,.0f}")
    col5.metric(
        "Average Discount Amount",
        f"${average_discount:,.2f}"
    )

    st.divider()

    # ---------------- Monthly trend analysis ----------------

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
            "Year or Month Number contains no valid data."
        )

    else:

        monthly["Month"] = (
            monthly["Year"].astype(int).astype(str)
            + "-"
            + monthly["Month Number"]
                .astype(int).astype(str).str.zfill(2)
        )

        left, right = st.columns(2)

        with left:

            sales_fig = px.line(
                monthly,
                x="Month",
                y="Sales",
                markers=True,
                hover_data=["Year", "Month Name"],
                title="Monthly Sales Trend",
            )

            sales_fig.update_yaxes(
                title="Sales ($)",
                tickprefix="$",
                tickformat=",.0f",
            )

            sales_fig.update_layout(
                xaxis_title="Month",
                hovermode="x unified",
            )

            st.plotly_chart(
                sales_fig,
                use_container_width=True,
            )

        with right:

            profit_fig = px.line(
                monthly,
                x="Month",
                y="Profit",
                markers=True,
                hover_data=["Year", "Month Name"],
                title="Monthly Profit Trend",
            )

            profit_fig.update_yaxes(
                title="Profit ($)",
                tickprefix="$",
                tickformat=",.0f",
            )

            profit_fig.update_layout(
                xaxis_title="Month",
                hovermode="x unified",
            )

            st.plotly_chart(
                profit_fig,
                use_container_width=True,
            )

    # ---------------- Data table ----------------

    with st.expander("View filtered sales data"):

        st.dataframe(
            filtered,
            use_container_width=True,
        )

    # ---------------- CSV export ----------------

    csv_data = filtered.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download filtered data as CSV",
        data=csv_data,
        file_name="filtered_sales_data.csv",
        mime="text/csv",
    )


# ============================================================
# 6. GROQ API CONFIGURATION
# ============================================================

# Streamlit Cloud uses Secrets; local development uses .env.
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:

    st.error(
        "Groq API key is missing. Add GROQ_API_KEY in "
        "Streamlit Cloud Settings > Secrets, or in your local .env file."
    )

    st.stop()

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=groq_api_key,
)

toolkit = SQLDatabaseToolkit(
    db=db,
    llm=model,
)

tools = toolkit.get_tools()


# ============================================================
# 7. SQL AGENT SYSTEM PROMPT
# ============================================================

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
4. Retrieve only the columns needed for the question.
5. Use SUM, AVG, MIN, MAX and COUNT when appropriate.
6. Use GROUP BY for category-level comparisons.
7. Use ORDER BY and LIMIT for ranking questions.
8. Limit detailed result sets to 10 rows by default.
9. Query only information from the database.
10. Do not create, alter, insert, update or delete records.
11. Inspect the schema when column names are uncertain.
12. Never invent numerical results or unsupported trends.

RESPONSE FORMAT:
1. Begin with a direct answer.
2. Present important figures clearly.
3. Provide two or three useful analytical observations when appropriate.
4. Base all findings on executed query results.
"""


@st.cache_resource
def get_agent():

    return create_agent(
        model=model,
        tools=tools,
        checkpointer=InMemorySaver(),
        system_prompt=system_prompt,
    )


agent = get_agent()


# ============================================================
# 8. APPLICATION TABS
# ============================================================

dashboard_tab, chat_tab = st.tabs(
    [
        "📊 Sales Dashboard",
        "💬 AI SQL Assistant",
    ]
)


# ============================================================
# 9. SALES DASHBOARD TAB
# ============================================================

with dashboard_tab:

    try:

        sales_df = load_sales_data(
            str(DB_PATH),
            DB_PATH.stat().st_mtime
        )

        show_sales_dashboard(sales_df)

    except Exception as exc:

        st.error("Unable to load the sales dashboard.")

        st.write(
            "Check the database, data.xlsx file, and column names."
        )

        with st.expander("View error details"):
            st.exception(exc)


# ============================================================
# 10. AI SQL ASSISTANT TAB
# ============================================================

with chat_tab:

    st.subheader(
        "📃 Data Analytics Hub | Text-to-SQL Business Intelligence"
    )

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous conversation.
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

        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
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
                                    "content": prompt,
                                }
                            ]
                        },
                        {
                            "configurable": {
                                "thread_id": st.session_state.thread_id
                            }
                        },
                    )

                    result = response["messages"][-1].content

                    if isinstance(result, list):

                        result = "\n".join(
                            block.get("text", "")
                            for block in result
                            if isinstance(block, dict)
                        )

                    st.markdown(result)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result,
                    })

                except Exception:

                    st.error(
                        "I couldn't answer that question. "
                        "Check the database, API key, and model configuration."
                    )
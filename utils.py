import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent

from fpdf import FPDF
from datetime import datetime



def combine_csvs(files):
    dfs = []
    for f in files:
        if f.name.endswith(".csv"):
            dfs.append(pd.read_csv(f))
        elif f.name.endswith(".xlsx"):
            dfs.append(pd.read_excel(f))
    return pd.concat(dfs, ignore_index=True)


def query_agent_groq(uploaded_files, query):
    df = combine_csvs(uploaded_files)

    llm = ChatGroq(
        temperature=0,
        model_name="llama3-70b-8192",  # Alternatives: mixtral-8x7b-32768, llama3-8b-8192
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=False,
        agent_type="openai-tools",  # Optional: can be omitted or changed
        handle_parsing_errors=True,
        allow_dangerous_code=True 
    )

    response = agent.run(query)

    fig = None
    if any(x in query.lower() for x in ["plot", "chart", "graph", "scatter", "hist", "visual"]):
        fig = generate_visual(df, query)

    return response, fig




def generate_visual(df, query):
    fig, ax = plt.subplots()
    query = query.lower()

    # --- Step 1: Auto-detect column types ---
    datetime_col = None
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    # Try parsing date columns
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                if df[col].notnull().sum() > 0:
                    datetime_col = col
                    break
            except:
                continue

    # Infer revenue if possible
    if 'revenue' not in df.columns and len(numeric_cols) >= 2:
        df['Revenue'] = df[numeric_cols[0]] * df[numeric_cols[1]]

    # --- Step 2: Visuals based on keywords in query ---

    if "monthly revenue" in query and datetime_col and 'Revenue' in df.columns:
        df['Month'] = df[datetime_col].dt.to_period('M')
        monthly = df.groupby('Month')['Revenue'].sum()
        monthly.plot(ax=ax, marker='o', title="Monthly Revenue Trend")
        ax.set_xlabel("Month")
        ax.set_ylabel("Revenue")
        ax.grid(True)

    elif "hist" in query or "distribution" in query:
        col = numeric_cols[0]
        df[col].hist(ax=ax, bins=20)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

    elif "correlation" in query or "heatmap" in query:
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="viridis", ax=ax)
        ax.set_title("Correlation Matrix")

    elif "scatter" in query and len(numeric_cols) >= 2:
        sns.scatterplot(data=df, x=numeric_cols[0], y=numeric_cols[1], ax=ax)
        ax.set_title(f"Scatter: {numeric_cols[0]} vs {numeric_cols[1]}")

    elif "top" in query and 'Revenue' in df.columns and categorical_cols:
        top = df.groupby(categorical_cols[0])['Revenue'].sum().sort_values(ascending=False).head(5)
        top.plot(kind='bar', ax=ax)
        ax.set_title(f"Top 5 {categorical_cols[0]} by Revenue")
        ax.set_ylabel("Revenue")

    else:
        return None  

    return fig




def save_chat_history(history, fig=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="CSV Chat Report", ln=True, align="C")
    pdf.ln(10)

    # Chat messages
    pdf.set_font("Arial", size=12)
    for msg in history:
        role = msg['role'].capitalize()
        content = msg['content']
        pdf.multi_cell(0, 10, f"{role}: {content}")
        pdf.ln(2)

    # Save visual as image if exists
    image_path = None
    if fig:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        image_path = f"temp_chart_{timestamp}.png"
        fig.savefig(image_path, bbox_inches="tight")
        pdf.ln(5)
        pdf.image(image_path, w=170)  # You can adjust width/height

    pdf_output = f"chat_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(pdf_output)

    # Read and return PDF binary
    with open(pdf_output, "rb") as f:
        pdf_data = f.read()

    # Clean up temp chart
    if image_path and os.path.exists(image_path):
        os.remove(image_path)
        os.remove(pdf_output)

    return pdf_data

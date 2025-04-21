import streamlit as st
import plotly.express as px
from col_datatype import detect_variable_type

def data_analysis_section(df):
    st.subheader("Data Analysis")

    column_types = {col: detect_variable_type(df, col) for col in df.columns}
    columns_with_types = [f"{col} ({col_type})" for col, col_type in column_types.items() if col_type != "Text"]
    col_name_map = {f"{col} ({col_type})": col for col, col_type in column_types.items() if col_type != "Text"}

    col1, col2 = st.columns(2)
    with col1:
        selected_x_display = st.selectbox("Select column for the X-axis", columns_with_types, index=None, placeholder="Select X-axis column...")
    with col2:
        selected_y_display = st.selectbox("Select column for the Y-axis", columns_with_types, index=None, placeholder="Select Y-axis column...")

    if selected_x_display and selected_y_display:
        selected_x_col = col_name_map.get(selected_x_display)
        selected_y_col = col_name_map.get(selected_y_display)

        type_x = detect_variable_type(df, selected_x_col)
        type_y = detect_variable_type(df, selected_y_col)

        chart_options = {
            ("Numeric", "Numeric"): ["Scatter", "Bar", "Line", "Bubble", "Histogram"],
            ("Numeric", "Categorical"): ["Scatter", "Bar"],
            ("Categorical", "Numeric"): ["Scatter", "Bar", "Funnel"],
            ("Categorical", "Categorical"): ["Scatter", "Bar", "Pie", "Treemap", "Funnel", "Sankey"]
        }

        available_charts = chart_options.get((type_x, type_y), []) or chart_options.get((type_y, type_x), [])

        if available_charts:
            chart_type = st.selectbox(f"Choose chart type for {selected_x_col} vs {selected_y_col}", available_charts)

            if chart_type == "Scatter":
                st.write(f"Scatter Plot between {selected_x_col} and {selected_y_col}")
                st.plotly_chart(px.scatter(df, x=selected_x_col, y=selected_y_col, color=selected_y_col), use_container_width=True)

            elif chart_type == "Bar":
                st.write(f"Bar Chart between {selected_x_col} and {selected_y_col}")
                st.plotly_chart(px.bar(df, x=selected_x_col, y=selected_y_col), use_container_width=True)

            elif chart_type == "Line":
                st.write(f"Line Chart between {selected_x_col} and {selected_y_col}")
                st.plotly_chart(px.line(df, x=selected_x_col, y=selected_y_col), use_container_width=True)

            elif chart_type == "Pie":
                st.write(f"Multiple Pie Charts grouped by {selected_x_col} and segmented by {selected_y_col}")
                fig = px.pie(
                    df,
                    names=selected_y_col,
                    facet_col=selected_x_col,
                    facet_col_wrap=3,
                    title=f"Pie charts for each {selected_x_col}"
                )
                fig.update_layout(
                    margin=dict(t=40, b=40),
                    height=((len(df[selected_x_col].unique()) // 3 + 1) * 300),
                    grid=dict(rows=1, columns=3),
                    uniformtext_minsize=12,
                    uniformtext_mode='hide'
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Histogram":
                st.write(f"Histogram for {selected_x_col}")
                st.plotly_chart(px.histogram(df, x=selected_x_col), use_container_width=True)

            elif chart_type == "Treemap":
                st.write(f"Treemap between {selected_x_col} and {selected_y_col}")
                st.plotly_chart(px.treemap(df, path=[selected_x_col, selected_y_col]), use_container_width=True)

            elif chart_type == "Funnel":
                st.write(f"Funnel Chart between {selected_x_col} and {selected_y_col}")
                st.plotly_chart(px.funnel(df, x=selected_x_col, y=selected_y_col), use_container_width=True)

            elif chart_type == "Bubble":
                st.write(f"Bubble Chart between {selected_x_col} and {selected_y_col}")
                st.plotly_chart(px.scatter(df, x=selected_x_col, y=selected_y_col, size=selected_y_col), use_container_width=True)

            elif chart_type == "Sankey":
                st.write(f"Sankey Chart between {selected_x_col} and {selected_y_col}")
                fig = px.sunburst(df, path=[selected_x_col, selected_y_col])
                st.plotly_chart(fig)
        else:
            st.warning("No suitable chart types available for the selected columns.")

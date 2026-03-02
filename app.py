import streamlit as st
from views.components import inject_css
from views import single_business, compare

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Review Aggregator",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔍 Review Aggregator")
    st.caption("Understand what your customers are really saying.")
    st.markdown("---")

    app_mode = st.radio(
        "Mode",
        ["🔍 Single Business", "⚖️ Compare Businesses"],
        key="app_mode_radio",
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown("---")

    # ── Session state init ──
    if "k_override" not in st.session_state:
        st.session_state.k_override = False
    if "compare_companies" not in st.session_state:
        st.session_state.compare_companies = []

    def on_slider_change():
        st.session_state.k_override = True

    # ── Single business sidebar ──
    if app_mode == "🔍 Single Business":
        n_clusters = st.slider(
            "Number of themes",
            min_value=2, max_value=8, value=5,
            key="n_clusters_slider",
            on_change=on_slider_change,
            help="Auto-selects optimal K via elbow analysis until you move this slider.",
        )
        if st.session_state.k_override:
            if st.button("↩️ Reset to auto", use_container_width=True):
                st.session_state.k_override = False

        st.markdown("---")
        st.markdown("**Data source**")
        data_source = st.radio(
            "Choose source",
            ["Sample data", "Scrape Trustpilot", "Upload CSV"],
            label_visibility="collapsed",
            key="data_source_radio",
        )

        trustpilot_url = None
        uploaded_file  = None
        max_pages      = 5
        business_name  = "Luigi's Bistro"
        n_clusters     = st.session_state.get("n_clusters_slider", 5)

        if data_source == "Sample data":
            business_name = st.text_input(
                "Business name", value="Luigi's Bistro", key="sample_business_name"
            )
        elif data_source == "Scrape Trustpilot":
            trustpilot_url = st.text_input(
                "Trustpilot URL or slug",
                placeholder="e.g. dominos.com",
                help="Full URL or just the slug",
                key="trustpilot_url_input",
            )
            max_pages = st.slider(
                "Max pages to scrape", 1, 20, 5,
                help="Each page = ~20 reviews",
                key="max_pages_slider",
            )
            st.caption("🕐 ~1 second per page")
        elif data_source == "Upload CSV":
            business_name = st.text_input(
                "Business name", placeholder="e.g. Mario's Pizzeria", key="csv_business_name"
            )
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="csv_uploader")
            st.caption("Required column: `review_text`\nOptional: `rating`")

    # ── Compare sidebar ──
    else:
        st.markdown("**Add a company**")
        compare_url = st.text_input(
            "Trustpilot URL or slug",
            placeholder="e.g. pizzahut.com",
            key="compare_url_input",
        )
        compare_pages = st.slider(
            "Max pages", 1, 10, 3,
            key="compare_pages_slider",
            help="Each page = ~20 reviews",
        )
        add_clicked = st.button("➕ Add company", use_container_width=True, type="primary")

        st.markdown("---")
        if st.session_state.compare_companies:
            st.markdown("**Companies added:**")
            for i, co in enumerate(st.session_state.compare_companies):
                ccol1, ccol2 = st.columns([3, 1])
                ccol1.caption(f"📍 {co['name']} ({co['review_count']} reviews)")
                if ccol2.button("✕", key=f"remove_{i}"):
                    st.session_state.compare_companies.pop(i)
                    st.rerun()

        if st.button("🗑️ Clear all", use_container_width=True,
                     disabled=not st.session_state.compare_companies):
            st.session_state.compare_companies = []
            st.rerun()


# ─────────────────────────────────────────────
# Route to view
# ─────────────────────────────────────────────

if app_mode == "🔍 Single Business":
    single_business.render(
        data_source=data_source,
        business_name=business_name,
        trustpilot_url=trustpilot_url,
        uploaded_file=uploaded_file,
        max_pages=max_pages,
        n_clusters=n_clusters,
    )
else:
    compare.render(
        add_clicked=add_clicked,
        compare_url=compare_url,
        compare_pages=compare_pages,
    )

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────

st.markdown("---")
st.caption("Review Aggregator · Built with Streamlit, scikit-learn & DistilBERT")
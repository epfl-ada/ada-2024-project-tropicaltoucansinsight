import streamlit as st
import os

# Define the folder paths
STATIC_FOLDER = 'static'

# Set page configuration
st.set_page_config(
    page_title="The YouTube Heavyweights: Entertainment vs. Music Face Off",
    layout="wide"
)

# Title and Abstract Section
st.title("The YouTube Heavyweights: Entertainment vs. Music Face Off")
st.markdown("""
Some claim the greatest 21st-century showdown was Floyd Mayweather versus Logan Paul, packed with stakes in marketing, \
            money, and public hype. We couldn’t disagree more: the real battle is Entertainment vs. Music on YouTube! \
            Leveraging the YouNiverse dataset, a massive collection of metadata covering 136k channels, 72.9M videos, and 2.8 years of \
            time series data on views and subscribers, we dive into YouTube’s top two categories, analyzing their reach through views, \
            subscriber counts, and strategic collaborations. Do entertainment creators ramp up content in December to maximize ad revenue? \
            Do music artists dominate the long game thanks to loyal fan bases? From seasonal trends to community dynamics, \
            we’ll explore how these giants shape and reshape their audiences. Get ready for a data showdown where each side fights for \
            the throne of influence, popularity, and engagement. Through time series analysis, hypothetical monetization, and network insights, \
            this is YouTube’s ultimate battle—where only one category can claim the crown in the world’s biggest digital arena!""")

st.markdown(""" ## The Face-Off of the Century Between the Two Biggest Contenders on YouTube: Music and Entertainment """)

st.markdown(""" Show pie-charts of different metrics to justify the choice of these two categories""")

st.markdown(""" ## Monetization Comparison for Music and Entertainment """)

st.markdown(""" Monetization results... """)

st.markdown(""" ## Collaboration Comparison Between Music and Entertainment""")

st.markdown(""" Collaboration results... """)

st.markdown(""" ## Popularity Consistency Comparison Between Music and Entertainment""")

st.markdown(""" Popularity Consistency results... """)

st.markdown(""" ## Diversity Comparison Between Music and Entertainment""")

st.markdown(""" Diversity results... """)


# Sidebar for Navigation
st.sidebar.title("Navigation")
sections = [
    "Introduction",
    "Research Questions & Methods",
    "Analysis and Results",
    "Website Features",
    "Conclusion & Future Work",
    "Meet the Team"
]
selected_section = st.sidebar.radio("Go to:", sections)

# Section logic
if selected_section == "Introduction":
    st.header("Introduction")
    st.markdown("""
    Welcome to our project page! Here, we explore the **ultimate showdown** between Entertainment and Music on YouTube. Through **data-driven analysis**, we aim to uncover:
    - Which category dominates monetization and audience engagement?
    - How seasonal trends and collaborations influence success?
    - What role community dynamics and diversity play?

    Let’s dive in!
    """)

elif selected_section == "Research Questions & Methods":
    st.header("Research Questions & Methods")

    # Collapsible sections for each research question
    with st.expander("1. Which category captures greater monetization potential?"):
        st.markdown("""
        - We introduce the **Monetization Potential (MP)** metric to assess earnings based on views, video duration, and ad-friendly periods.
        - Seasonal trends are analyzed for peaks (e.g., December) and troughs (e.g., summer).
        """)

    with st.expander("2. Which category offers broader diversity in content types?"):
        st.markdown("""
        - We cluster content themes using ML techniques like **SpaCy** and **RoBERTa**.
        - Diversity is correlated with engagement metrics, revealing potential advantages in audience retention.
        """)

    with st.expander("3. Which category leverages collaborations more effectively?"):
        st.markdown("""
        - Collaboration patterns are identified through text mining on titles and descriptions.
        - Impact on viewership and reach is compared across categories.
        """)

    with st.expander("4. Which category maintains consistent popularity?"):
        st.markdown("""
        - We analyze short-term vs. long-term trends in viewership.
        - Outliers and loyalty metrics provide insights into sustained audience interest.
        """)

    with st.expander("5. Which category uses seasonal release patterns effectively?"):
        st.markdown("""
        - Time-series data reveals whether Entertainment or Music capitalizes on high-viewership periods like December.
        """)

    with st.expander("6. How do major releases affect growth?"):
        st.markdown("""
        - Case studies of significant releases (e.g., albums or viral projects) highlight subscriber growth and engagement boosts.
        """)

    with st.expander("7. Optional: What are the differences in community dynamics?"):
        st.markdown("""
        - Using PySpark, we explore community structures, overlaps, and evolution between Music and Entertainment audiences.
        """)

elif selected_section == "Analysis and Results":
    st.header("Analysis and Results")
    st.markdown("""
    ### Key Findings
    - **Monetization Potential**: Entertainment sees peaks during holiday seasons, while Music maintains steady long-term monetization.
    - **Content Diversity**: Entertainment shows greater format diversity, leading to higher short-term engagement spikes.
    - **Collaborations**: Music channels leverage collaborations more effectively, as evidenced by their consistent viewership boosts.
    - **Seasonal Patterns**: Entertainment capitalizes on holiday seasons like December, while Music benefits from album releases.
    """)

    # Placeholder for visuals
    st.subheader("Visualizations")
    st.markdown("Below are some key visual insights:")
    st.markdown("TBD...")

elif selected_section == "Website Features":
    st.header("Website Features")
    st.markdown("""
    This website provides:
    - Interactive visualizations of key trends in monetization, engagement, and viewership.
    - A detailed comparison of Entertainment and Music channels using filters for custom exploration.
    - Real-time insights into seasonal and collaborative dynamics.
    """)

elif selected_section == "Conclusion & Future Work":
    st.header("Conclusion & Future Work")
    st.markdown("""
    ### Conclusion
    - Our analysis reveals distinct strengths for both categories, with **Entertainment** dominating seasonal trends and **Music** excelling in collaborations.

    ### Future Work
    - Expand the analysis to include other categories like Gaming or Education.
    - Dive deeper into community dynamics using advanced network analysis.
    """)

elif selected_section == "Meet the Team":
    st.header("Meet the Team")
    st.markdown("""
        Behind the scenes of this project, we are a team of **data gladiators** from EPFL, uniting the precision of **Physics**, the logic of **Computer Science**, and the wizardry of **Data Science**. Together, we’ve forged an unstoppable alliance to crack YouTube’s biggest mysteries—because why just *study* science when you can make it viral?
        """)

    # Team Members Section
    col1, col2, col3, col4 = st.columns(4)

    # Team Member 1: Sylvain
    with col1:
        st.image(
            os.path.join(STATIC_FOLDER, "placeholder_sylvain.jpg"),
            width=150,
            use_container_width=True
        )
        st.caption("<center>Sylvain<br>Physics, EPFL<br><i>Data Analyst</i>", unsafe_allow_html=True)

    # Team Member 2: Max
    with col2:
        st.image(
            os.path.join(STATIC_FOLDER, "placeholder_max.jpg"),
            width=150,
            use_container_width=True
        )
        st.caption("<center>Max<br>Physics, EPFL<br><i>Data Analyst</i>", unsafe_allow_html=True)

    # Team Member 3: Timothée
    with col3:
        st.image(
            os.path.join(STATIC_FOLDER, "placeholder_timothee.jpg"),
            width=150,
            use_container_width=True
        )
        st.caption("<center>Timothée<br>Physics, EPFL<br><i>Data Analyst</i>", unsafe_allow_html=True)

    # Team Member 4: Jérémy
    with col4:
        st.image(
            os.path.join(STATIC_FOLDER, "placeholder_jeremy.jpg"),
            width=150,
            use_container_width=True
        )
        st.caption("<center>Jérémy<br>Data Science, EPFL<br><i>Data Analyst</i>", unsafe_allow_html=True)


# Footer with link to "Meet the Team"
st.markdown("---")
st.markdown("Made with ❤️ by **Team Tropical Toucans Insight**. CS401 Applied Data Analysis course project conducted in Fall '24 at EPFL, Switzerland.")
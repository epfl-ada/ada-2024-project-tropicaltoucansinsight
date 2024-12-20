import streamlit as st
import os
from streamlit_pdf_viewer import pdf_viewer
import io
import pickle
import base64

# Define the folder paths
STATIC_FOLDER = 'static'
FIGURES_PDF = 'figures\pdf'
FIGURES_PKL = 'figures\pickle'

# Set page configuration
st.set_page_config(
    page_title="The YouTube Heavyweights: Entertainment vs. Music Face Off",
    layout="wide"
)

def get_base64_encoded_image(image_path):
    with open('figures\\Box_fight_Image.jpg', "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Load the local image and encode it
image_path = "banner.jpg"  # Replace with the path to your image file
encoded_image = get_base64_encoded_image(image_path)

st.markdown(
    f"""
    <style>
    .top-banner {{
        position: relative;
        width: 100%;
        height: 800px;  /* Adjust this height for your image */
        background-image: url('data:image/jpeg;base64,{encoded_image}');
        background-size: cover;  /* Ensure the image fits without cutting off */
        background-repeat: no-repeat;
        background-position: center;
        display: flex;
        justify-content: center;
        align-items: center;
        opacity: 1; /* No fading for the image */
    }}

     .top-banner::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(0, 0, 0, 0.5);  /* Dark overlay */
        z-index: 1;
    }}

    .top-banner h1 {{
        position: relative;
        z-index: 2;  /* Ensure the title appears above the overlay */
        margin: 0;
        padding: 0;
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        font-family: 'Arial', sans-serif;
        color: white;  /* White color for bright text */
        text-shadow: 3px 3px 20px rgba(0, 0, 0, 0.7);  /* Bright text-shadow for better visibility */
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Render the top banner
st.markdown(
    """
    <div class="top-banner">
        <h1>The YouTube Heavyweights: Entertainment vs. Music Face Off</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# Title and Abstract Section
st.write("")
st.markdown('<div style="text-align: justify;"> Some claim the greatest 21st-century showdown was Floyd Mayweather versus Logan Paul, packed with stakes in marketing, \
            money, and public hype. We couldn’t disagree more: the real battle is Entertainment vs. Music on YouTube! \
            Leveraging the YouNiverse dataset, a massive collection of metadata covering 136k channels, 72.9M videos, and 2.8 years of \
            time series data on views and subscribers, we dive into YouTube’s top two categories, analyzing their reach through views, \
            subscriber counts, and strategic collaborations. Do entertainment creators ramp up content in December to maximize ad revenue? \
            Do music artists dominate the long game thanks to loyal fan bases? From seasonal trends to community dynamics, \
            we’ll explore how these giants shape and reshape their audiences. Get ready for a data showdown where each side fights for \
            the throne of influence, popularity, and engagement. Through time series analysis, hypothetical monetization, and network insights, \
            this is YouTube’s ultimate battle—where only one category can claim the crown in the world’s biggest digital arena! </div>', unsafe_allow_html=True)

st.markdown(""" ## The Face-Off of the Century Between the Two Biggest Contenders on YouTube: Music and Entertainment """)

fig1_path = os.path.join(FIGURES_PKL, "pie_chart.pkl")
with open(fig1_path, "rb") as f:
    fig1 = pickle.load(f)
svg_buffer = io.StringIO()
fig1.savefig(svg_buffer, format='svg', bbox_inches='tight')
svg_buffer.seek(0)
svg_data = svg_buffer.getvalue()
svg_data_scaled = svg_data.replace('<svg ', '<svg style="width:100%; height:100%;" ')
st.components.v1.html(f"<div>{svg_data_scaled}</div>", height=600, scrolling=True)

st.caption("Pie Chart", unsafe_allow_html=True)

fig1_path = os.path.join(FIGURES_PKL, "pie_chart_1.pkl")
with open(fig1_path, "rb") as f:
    fig1 = pickle.load(f)
svg_buffer = io.StringIO()
fig1.savefig(svg_buffer, format='svg', bbox_inches='tight')
svg_buffer.seek(0)
svg_data = svg_buffer.getvalue()
svg_data_scaled = svg_data.replace('<svg ', '<svg style="width:100%; height:100%;" ')
st.components.v1.html(f"<div>{svg_data_scaled}</div>", height=600, scrolling=True)


st.caption("Pie Chart 1", unsafe_allow_html=True)

fig1_path = os.path.join(FIGURES_PKL, "pie_chart_2.pkl")
with open(fig1_path, "rb") as f:
    fig1 = pickle.load(f)
svg_buffer = io.StringIO()
fig1.savefig(svg_buffer, format='svg', bbox_inches='tight')
svg_buffer.seek(0)
svg_data = svg_buffer.getvalue()
svg_data_scaled = svg_data.replace('<svg ', '<svg style="width:100%; height:100%;" ')
st.components.v1.html(f"<div>{svg_data_scaled}</div>", height=600, scrolling=True)

st.caption("Pie Chart 2", unsafe_allow_html=True)

st.markdown(""" Show pie-charts of different metrics to justify the choice of these two categories""")

st.markdown(""" ## Monetization Comparison for Music and Entertainment """)

st.markdown(""" Monetization results... """)

st.markdown(""" ## Collaboration Comparison Between Music and Entertainment""")

st.markdown(""" Collaboration results... """)

st.markdown(""" ## Popularity Consistency Comparison Between Music and Entertainment""")

st.markdown("""<div style="text-align: justify;">The fighters have already gone through half of the number of rounds and are starting to get tired.\
            The match has been close so far. Who will be the winner? </div>""", unsafe_allow_html=True)
st.write("")
st.markdown("""<div style="text-align: justify;">This next round will oppose Music and Entertainment on how well they maintain popularity on YouTube. As soon as the \
            timer starts, Music and Entertainment both display the average number of views their channels get in one week (\(\Delta\)Views), as well\
            as the average number of new videos published per week (\(\Delta\)Videos). </div>""", unsafe_allow_html=True)
st.write("")


# Figures side by side for time-series of overall delta_views mean and delta_videos mean
fig_path_1 = os.path.join(FIGURES_PKL, "general_rolling_mean.pkl")
with open(fig_path_1, "rb") as f:
    fig1 = pickle.load(f)
svg_buffer = io.StringIO()
fig1.savefig(svg_buffer, format='svg', bbox_inches='tight')
svg_buffer.seek(0)
svg_data1 = svg_buffer.getvalue()
svg_data_scaled1 = svg_data1.replace('<svg ', '<svg style="width:100%; height:auto;" ')

fig_path_2 = os.path.join(FIGURES_PKL, "general_rolling_mean_2.pkl")
with open(fig_path_2, "rb") as f:
    fig2 = pickle.load(f)
svg_buffer = io.StringIO()
fig2.savefig(svg_buffer, format='svg', bbox_inches='tight')
svg_buffer.seek(0)
svg_data2 = svg_buffer.getvalue()
svg_data_scaled2 = svg_data2.replace('<svg ', '<svg style="width:100%; height:auto;" ')

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f'<div style="text-align: center;">{svg_data_scaled1}</div>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f'<div style="text-align: center;">{svg_data_scaled2}</div>',
        unsafe_allow_html=True,
    )
st.write("")
st.markdown("""<div style="text-align: justify;">The result is immediately clear: as of approximately 2017, Entertainment channels dominate with higher average number of\
            new views and new videos each week. So at a first glance, it would seem as though Entertainment has higher weekly average popularity.\
            However, one can wonder whether this occurs only because more videos are published, or if Entertainment actually has higher popularity\
            than Music. </div>""", unsafe_allow_html=True)
st.write("")
st.markdown("""<div style="text-align: justify;">To make this even more interesting, Music and Entertainment now show off their top assets: the channels from these two categories\
            that together amass 75% of the total number of subscribers. </div>""", unsafe_allow_html=True)
st.write("")

# Figures side by side for time-series of filtered delta_views and delta_videos mean
fig_path_1 = os.path.join(FIGURES_PKL, "general_rolling_mean_3.pkl")
with open(fig_path_1, "rb") as f:
    fig1 = pickle.load(f)
svg_buffer = io.StringIO()
fig1.savefig(svg_buffer, format='svg', bbox_inches='tight')
svg_buffer.seek(0)
svg_data1 = svg_buffer.getvalue()
svg_data_scaled1 = svg_data1.replace('<svg ', '<svg style="width:100%; height:auto;" ')

fig_path_2 = os.path.join(FIGURES_PKL, "general_rolling_mean_5.pkl")
with open(fig_path_2, "rb") as f:
    fig2 = pickle.load(f)
svg_buffer = io.StringIO()
fig2.savefig(svg_buffer, format='svg', bbox_inches='tight')
svg_buffer.seek(0)
svg_data2 = svg_buffer.getvalue()
svg_data_scaled2 = svg_data2.replace('<svg ', '<svg style="width:100%; height:auto;" ')

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f'<div style="text-align: center;">{svg_data_scaled1}</div>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f'<div style="text-align: center;">{svg_data_scaled2}</div>',
        unsafe_allow_html=True,
    )
st.write("")

st.markdown("""<div style="text-align: justify;">This time around, the advantage is less pronounced: Music's top assets seem to rival those of Entertainment for the average number\
            of new views each week, even though, on average, the top channels of Entertainment upload videos much more frequently than Music.\
            This round has started off in an exciting fashion. Let's see which contender still has tricks up their sleeve. </div>""", unsafe_allow_html=True)
st.write("")
st.markdown("""<div style="text-align: justify;">We now ask of the fighters to present how the average number of views evolves after the upload of a single video.\
            They will show off different time frames after a video has been uploaded where no new video is published, so that the viewer\
            can get an unconfounded look at how popularity decays. </div>""", unsafe_allow_html=True)
st.write("")
# Overall time evolution of delta_views
fig1_path = os.path.join(FIGURES_PKL, "Overall_Time_Evolution_Delta_Views.pkl")
with open(fig1_path, "rb") as f:
    fig1 = pickle.load(f)
svg_buffer = io.StringIO()
fig1.savefig(svg_buffer, format='svg', bbox_inches='tight')
svg_buffer.seek(0)
svg_data = svg_buffer.getvalue()
svg_data_scaled = svg_data.replace('<svg ', '<svg style="width:100%; height:100%;" ')
st.components.v1.html(f"<div>{svg_data_scaled}</div>", height=900, scrolling=True)
st.write("")
st.markdown("""<div style="text-align: justify;">What an interesting turn of events! After the upload of a video, Music channels have a higher average number of new views per week, \
            for all time frames. This is a surprising result, as before we saw that Entertainment channels had a higher mean number of views \
            each week. Therefore, when a new video is published, the popularity response is on average higher for Music. We can also observe that \
            the decreasing trends seem of similar shape, with some strange peaks caused by outliers in Entertainment channels at longer \
            time frames. </div>""", unsafe_allow_html=True)
st.write("")
st.markdown("""<div style="text-align: justify;">This round is now well off and we are seeing clear signs of fatigue in our two opponents. It seems as though Music has taken over, \
            but victory can not yet be declared. The next big event is about return times: that is, the time a spike in new views in a \
            channel\'s time evolution takes to return to a baseline calculated prior to this spike. The contenders will display the \
            average return times for each of their channels, as well as all their return times independently of the channel they belong too. </div>""", unsafe_allow_html=True)
st.write("")

# Figures side by side for overall return times
fig_path_1 = os.path.join(FIGURES_PKL, "overall_channel_avg_return_times.pkl")
with open(fig_path_1, "rb") as f:
    fig1 = pickle.load(f)
svg_buffer = io.StringIO()
fig1.savefig(svg_buffer, format='svg', bbox_inches='tight')
svg_buffer.seek(0)
svg_data1 = svg_buffer.getvalue()
svg_data_scaled1 = svg_data1.replace('<svg ', '<svg style="width:108%; height:auto;" ')

fig_path_2 = os.path.join(FIGURES_PKL, "overall_return_times.pkl")
with open(fig_path_2, "rb") as f:
    fig2 = pickle.load(f)
svg_buffer = io.StringIO()
fig2.savefig(svg_buffer, format='svg', bbox_inches='tight')
svg_buffer.seek(0)
svg_data2 = svg_buffer.getvalue()
svg_data_scaled2 = svg_data2.replace('<svg ', '<svg style="width:93%; height:auto;" ')

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f'<div style="text-align: center;">{svg_data_scaled1}</div>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f'<div style="text-align: center;">{svg_data_scaled2}</div>',
        unsafe_allow_html=True,
    )

st.write("")
st.markdown("""<div style="text-align: justify;">It would now seem as though Music has taken a definite lead in this round. We can clearly observe that Music channels \
            have a higher density at longer return times, be it for channel-averaged values or overall. For Music to confirm its advantage, \
            it also needs to demonstrate resilience by having spikes decay slower on average than Entertainment. This is what we see next: \
            both Music and Entertainment will display the average rate at which a peak in new views decays until it reaches the \
            baseline value. Once again, they will show results for channel averages and overall decay rates. </div>""", unsafe_allow_html=True)
st.write("")
# Figures side by side for overall decay rates
fig_path_1 = os.path.join(FIGURES_PKL, "overall_channel_avg_decay_rates.pkl")
with open(fig_path_1, "rb") as f:
    fig1 = pickle.load(f)
svg_buffer = io.StringIO()
fig1.savefig(svg_buffer, format='svg', bbox_inches='tight')
svg_buffer.seek(0)
svg_data1 = svg_buffer.getvalue()
svg_data_scaled1 = svg_data1.replace('<svg ', '<svg style="width:108%; height:auto;" ')

fig_path_2 = os.path.join(FIGURES_PKL, "overall_decay_rates.pkl")
with open(fig_path_2, "rb") as f:
    fig2 = pickle.load(f)
svg_buffer = io.StringIO()
fig2.savefig(svg_buffer, format='svg', bbox_inches='tight')
svg_buffer.seek(0)
svg_data2 = svg_buffer.getvalue()
svg_data_scaled2 = svg_data2.replace('<svg ', '<svg style="width:93%; height:auto;" ')

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f'<div style="text-align: center;">{svg_data_scaled1}</div>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f'<div style="text-align: center;">{svg_data_scaled2}</div>',
        unsafe_allow_html=True,
    )

st.write("")
st.markdown("""<div style="text-align: justify;">It seems as though we can predict where this round is going: Music channels have smaller decay rates than Entertainment channels, \
            as the density for Entertainment is greater for higher decay rates. This now means that Music has confirmed that its spikes in popularity\
            last longer and decay more slowly than Entertainment\'s popularity spikes. </div>""", unsafe_allow_html=True)
st.write("")
st.markdown("""<div style="text-align: justify;">But wait! What if this was all a big fluke? It could be possible that Music has longer return times and smaller decay rates \
            just because its spikes are of greater magnitude. Therefore, this would show that there are bigger hype fluctuations \
            for Music channels (this was already the case above), but not that this new popularity is maintained more consistently. Just \
            before the timer ends, Music and Entertainment will display the height of their peaks above the baseline value. This will \
            most definitely determine the outcome of this round! </div>""", unsafe_allow_html=True)
st.write("")

# Figure of overall peak heights
fig_path_1 = os.path.join(FIGURES_PKL, "Peak_Heights_Overall.pkl")
with open(fig_path_1, "rb") as f:
    fig1 = pickle.load(f)
svg_buffer = io.StringIO()
fig1.savefig(svg_buffer, format='svg', bbox_inches='tight')
svg_buffer.seek(0)
svg_data1 = svg_buffer.getvalue()
svg_data_scaled1 = svg_data1.replace('<svg ', '<svg style="width:100%; height:auto;" ')
st.components.v1.html(f"<div>{svg_data_scaled1}</div>", height=500, scrolling=True)

st.write("")
st.markdown("""<div style="text-align: justify;">Well, it seems as though we have a winner for this round! The distribution of the peak heights is very similar between Music\
            and Entertainment, where the difference in frequencies can be explained simply by a smaller number of spikes in Music. Hence, \
            we can conclude that Music maintains popularity better than Entertainment. </div>""", unsafe_allow_html=True)
st.write("")
st.markdown("""<div style="text-align: justify;">The third round has now ended and the winner has been declared as Music. In this case, domination was clear: from higher \
            responses after the upload of a new video, to longer return times and shorter decay rates. We therefore \
            give 10 points to Music and 8 points to Entertainment! </div>""", unsafe_allow_html=True)
st.write("")

st.markdown("""<div style="text-align: justify;">While we wait for the last round to start as we give our contenders a small break to rest and tend to their wounds, we invite\
            the spectator to take a look at the time evolution of $$\Delta$$Views for a few of the most famous assets of the Music category.\
            These time-series highlight peaks, return times, as well as display when new videos were uploaded. </div>""", unsafe_allow_html=True)

st.write("")
# Select time-series figures
def get_svg_from_figure(fig_path):
    with open(fig_path, "rb") as f:
        fig = pickle.load(f)
    svg_buffer = io.StringIO()
    fig.savefig(svg_buffer, format="svg", bbox_inches="tight")
    svg_buffer.seek(0)
    svg_data = svg_buffer.getvalue()
    # Ensure the SVG fits the container
    svg_data_scaled = svg_data.replace(
        '<svg ', '<svg style="width:100%; height:auto; display:block; margin:auto;" '
    )
    return svg_data_scaled

# Paths to figures
figure_paths = {
    "Justin Bieber": os.path.join(FIGURES_PKL, "Justin_Bieber_time_series.pkl"),
    "Ed Sheeran": os.path.join(FIGURES_PKL, "Ed_Sheeran_time_series.pkl"),
    "Coldplay": os.path.join(FIGURES_PKL, "Coldplay_time_series.pkl")
}

# User selection
selected_figure = st.selectbox("Choose a figure to display:", list(figure_paths.keys()))

# Display the selected figure
if selected_figure:
    svg_data = get_svg_from_figure(figure_paths[selected_figure])
    st.markdown(f'<div style="text-align: center;">{svg_data}</div>', unsafe_allow_html=True)


st.markdown(""" ## Diversity Comparison Between Music and Entertainment""")

st.markdown(""" Diversity results... """)

st.markdown(""" ## Conclusion """)

st.markdown(""" Something to conclude """)



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
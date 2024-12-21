import streamlit as st
import os
import io
import pickle
import base64
import pandas as pd
from streamlit_pdf_viewer import pdf_viewer

# Define the folder paths
STATIC_FOLDER = 'static'
FIGURES_PDF = os.path.join('figures','pdf')
FIGURES_PKL = os.path.join('figures','pickle')

# Set page configuration
st.set_page_config(
    page_title="The YouTube Heavyweights: Entertainment vs. Music Face Off",
    layout="wide"
)

def get_base64_encoded_image(image_path):
    with open(os.path.join('static','intro.jpg'), "rb") as img_file:
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
st.header("Introduction")
    
st.markdown('<div style="text-align: justify;"> Some claim the greatest 21st-century showdown was Floyd Mayweather versus Logan Paul, packed with stakes in marketing, \
            money, and public hype. We couldn’t disagree more: the real battle is Entertainment vs. Music on YouTube! \
            Leveraging the YouNiverse dataset, a massive collection of metadata covering 136k channels, 72.9M videos, and 2.8 years of \
            time series data on views and subscribers, we dive into YouTube’s top two categories, analyzing their reach through views, \
            subscriber counts, and strategic collaborations. Do entertainment creators ramp up content in December to maximize ad revenue? \
            Do music artists dominate the long game thanks to loyal fan bases? From seasonal trends to community dynamics, \
            we’ll explore how these giants shape and reshape their audiences. Get ready for a data showdown where each side fights for \
            the throne of influence, popularity, and engagement. Through time series analysis, hypothetical monetization, collaboration statistics, \
            consistency of popularity and diversity, this is YouTube’s ultimate battle—where only one category can claim the crown in the world’s biggest digital arena! </div>', unsafe_allow_html=True)
st.write("")

st.markdown("""<div style="text-align: justify;">To determine whether Music or Entertainment dominates the YouTube scene, we have organized our analysis \
            as a boxing match with four different rounds. As in a typical boxing match, points are awarded based on the winner of each round. The \
            winner always get 10 points. The loser gets 9 points if the round was close, 8 if there was clear domination and 7 for extreme cases.\
            At the end of the allotted number of round, the winner is the one with the highest number of points. The four rounds of this showdown \
            will be Potential Monetization, Collaborations, Consistency of Popularity and Diversity of Content. Now sit back, relax, and \
            enjoy the show.</div>""", unsafe_allow_html=True)


# Sidebar for Navigation
st.sidebar.title("Navigation")
sections = [
    "Analysis and Results",
    "Conclusion",
    "Meet the Team"
]
selected_section = st.sidebar.radio("Go to:", sections)


if selected_section == "Analysis and Results":
    st.markdown(""" ## The Face-Off of the Century Between the Two Biggest Contenders on YouTube: Music and Entertainment """)

    st.markdown("""<div style="text-align: justify;">To first justify the choice of Music and Entertainment categories as being the biggest and therefore the most interesting categories\
            to compare on YouTube, we give some statistics about how the number of channels and subscribers is distributed over the different \
            categories in the YouNiverse dataset.</div>""", unsafe_allow_html=True)
    
    st.write("")

    fig1_path = os.path.join(FIGURES_PKL, "pie_chart.pkl")
    with open(fig1_path, "rb") as f:
        fig1 = pickle.load(f)
    svg_buffer = io.StringIO()
    fig1.savefig(svg_buffer, format='svg', bbox_inches='tight')
    svg_buffer.seek(0)
    svg_data = svg_buffer.getvalue()
    svg_data_scaled = svg_data.replace('<svg ', '<svg style="width:100%; height:100%;" ')
    st.components.v1.html(f"<div>{svg_data_scaled}</div>", height=600, scrolling=True)


    fig1_path = os.path.join(FIGURES_PKL, "pie_chart_1.pkl")
    with open(fig1_path, "rb") as f:
        fig1 = pickle.load(f)
    svg_buffer = io.StringIO()
    fig1.savefig(svg_buffer, format='svg', bbox_inches='tight')
    svg_buffer.seek(0)
    svg_data = svg_buffer.getvalue()
    svg_data_scaled = svg_data.replace('<svg ', '<svg style="width:100%; height:100%;" ')
    st.components.v1.html(f"<div>{svg_data_scaled}</div>", height=600, scrolling=True)

    st.markdown("""<div style="text-align: justify;">We observe that Music and Entertainment together represent approximately 35% of total number of channels in the dataset and \
            amass approximately 45% of the number of subscribers. Therefore, these are clearly the two heavyweight categories of the YouNiverse \
            dataset and will most definitely deliver an exciting fight. Let the match begin!</div>""", unsafe_allow_html=True)

    st.markdown(""" ## Monetization Comparison for Music and Entertainment """)

    st.markdown("""<div style="text-align: justify;">As an adventurous viewer venturing into the vast realm of YouTube content, you might wonder: Where does the gold lie—in the vibrant rhythm of "Music" or the diverse storytelling of "Entertainment"? The bell rings for the first round of this showdown, focused on Monetization Potential (MP).\
    Our Monetization Potential (MP) is a speculative yet insightful metric that estimates earnings potentially generated by ads integrated into videos, combining key factors such as views, video duration, and ad-friendly periods.\
    The stakes are high; understanding which category yields greater revenue is crucial for creators, advertisers, and platform strategists. Let’s jump into the ring and see who takes the lead.</div>""", unsafe_allow_html=True)
    
    st.markdown("""<div style="text-align: justify;">In one corner, we have Music, a compact powerhouse known for attracting massive audiences with its consumable format. Yet, its shorter durations make ad integration a challenge.\
                    In the opposite corner stands Entertainment, a diverse category encompassing challenges, commentary, and storytelling. With videos often exceeding ten minutes, this category has the potential for multiple ad slots.</div>""", unsafe_allow_html=True)
    st.write("")
    st.markdown("""<div style="text-align: justify;">In this opening round, a critical factor comes into play: look, in the first of the following plots, at the reach of the Entertainment category compared to Music. While music videos typically hover around 3 minutes in length, Entertainment videos span a broader range—some align with music’s brevity, but a significant proportion cluster around 11 minutes. Concerning the distribution of views of the second plot, the music category has the upper hand with a bigger tail on its distribution, showing that it has more highly viewed videos.</div>""", unsafe_allow_html=True)
    st.write("")


    # Figures side by side for overall decay rates
    fig_path_1 = os.path.join(FIGURES_PDF, "Distribution_of_Duration_mieux.pdf")
    pdf_viewer(fig_path_1, width=900, height=450)


    fig_path_2 = os.path.join(FIGURES_PDF, "Distribution_of_Number of Views.pdf")
    pdf_viewer(fig_path_2, width=900, height=450)


    st.markdown("""<div style="text-align: justify;">As the contenders size each other up, let me ask you a question: do you know how ad monetization works on YouTube? This round will be easier to follow if you understand that. Here’s why video duration matters. When a YouTuber monetizes its channel, it allows ads to be inserted in their videos, splitting the revenue with the platform. The longer the video, the more ad slots can be included—beyond just pre-roll and post-roll ads. Broadly speaking, every 8 minutes of video allows for an additional ad slot, a key factor accounted for in our calculations.\
    But ads only generate revenue if viewers watch them. Entertainment videos are expected to command more viewer attention than music, which is often played in the background. This is why the CPM (Cost Per Mille, or revenue per thousand views) is inherently higher for Entertainment than for Music.\
    In this round, Entertainment enters with an advantage: a higher CPM and greater reach (longer durations). But don’t count Music out just yet! The Music category has some of the most viewed videos on the platform—a clear crowd favorite in this matchup!</div>""", unsafe_allow_html=True)
    """"""

    fig1_path = os.path.join(FIGURES_PDF, "Distribution_of_Estimated Revenue (USD)_mieux.pdf")
    pdf_viewer(fig1_path, width=900, height=450)

    st.write("")
    st.markdown("""<div style="text-align: justify;">Using the MP metric across thousands of videos, the results come in:</div>""", unsafe_allow_html=True)
    st.markdown("""<div style="text-align: justify;">⁠Entertainment Takes the Round—Barely: Entertainment edges out Music with slightly higher MP values. We can distinguish how the two characteristic durations of entertainment videos, that we underlined before, provide room for multiple ad slots, giving it the upper hand.\
                ⁠Ad Integration as the Deciding Factor: Entertainment’s flexibility in incorporating pre-roll, post-roll but above all mid-roll ads secures its narrow victory. In contrast, Music’s reliance on pre-roll ads limits its monetization options.\
                The difference may be slim, but the bell signals the end of Round 1, with the judges claiming the win of Entertainment category 10 to 9!</div>""", unsafe_allow_html=True)

    st.markdown("""<div style="text-align: justify;">As the two contenders walk back to their corners, we are already looking ahead to the next rounds…
                                Indeed, the estimation of the monetization here lacked precision: the numbers of views were too close, and the bigger durations of a significant part of the entertainment videos have been the only parameter to really influence the decision. Indeed, in order to truly transform this monetization potential and, even more, to be a relevant and legitimate content on the YouTube platform, a category needs to generate engagement among its viewers so that they watch the videos for as long as possible and come back to rewatch them or to watch new content. \
                                The fight is far from over. Round 1 has set the tone, but deeper questions loom:</div>""", unsafe_allow_html=True)
    st.markdown("""<div style="text-align: justify;">-Popularity & consistency in time: Can Music be more consistent in popularity than Entertainment?</div>""", unsafe_allow_html=True)
    st.markdown("""<div style="text-align: justify;">-Content Variety: How does the diversity of entertainment content broaden its appeal?</div>""", unsafe_allow_html=True)
    st.markdown("""<div style="text-align: justify;">These battles will play out in the upcoming rounds as we dissect engagement patterns, content strategies, etc…</div>""", unsafe_allow_html=True)
    st.markdown("""<div style="text-align: justify;">     -Collaborations: Do the players in these categories know how to surround themselves and pull each other up, and in what way?</div>""", unsafe_allow_html=True)
    st.write("")
    st.markdown(""" ## Collaboration Comparison Between Music and Entertainment""")

    st.markdown("""<div style="text-align: justify;">
    After a tight first round, the two contenders return to their corners. The audience is buzzing with excitement and action: users click, share, comment, repost... In the middle of this digital frenzy, a few words emerge clearly from both sides: “more” … “stronger” … “feat” … “collaborate!” An unexpected call for collaboration? This could be Music’s chance to close the gap? 

The fight resumes, and it’s clear both contenders have taken the message to heart. A shift in strategy is evident as both categories bring fresh energy to the battle.  But did the coaches give good advice? Is collaboration truly beneficial? To answer this, we first investigate whether collaboration positively impacts video performance metrics—views, likes, and dislikes—in each category.  However, before diving into the data, we need to detect which videos are collaborations. Titles play a crucial role in this process, as they are often the first point of interaction for viewers and can significantly influence engagement. Words like _feat_, _ft_, and _collaboration_ in titles are strong indicators of joint efforts between creators and may even attract more attention by highlighting partnerships. To identify collaborative videos, we use a keyword-based method, scanning titles for terms that commonly indicate collaboration. This approach enables us to isolate and analyse these videos effectively.  Our analysis then compares the performance of collaborative and non-collaborative videos through visual methods such as histograms and boxplots. Additionally, we apply statistical tests to determine whether the observed differences are significant. 

To understand how collaboration could tip the scales for leading channels, we use our  data, focusing on the videos with the highest views that collectively contribute to 95% of the total views in each category. The results of the filtering and collaboration detection are displayed in the table bellow. 
    <\div>""", unsafe_allow_html=True)


    data = { "Category": ["Music", "Entertainment"], "Original Number of Videos": [8197981, 12015676], "Top Videos (95%)": [541108, 1782414], "Fraction of Videos in the Top (%)": [6.60, 14.83], "Number of Detected Collaborations": [67893, 46357], "Fraction of Collaborations in the Top (%)": [12.55, 2.60], }
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)


    st.markdown("""<div style="text-align: justify;">
    Out of the total number of videos, only a small fraction accounts for 95% of the total views. In the Music category, just 6.60% of videos (541,108 out of 8,197,981) dominate viewership. This highlights a strong reliance on a limited number of highly popular hits. In contrast, the Entertainment category requires 14.83% of its videos (1,782,414 out of 12,015,676) to reach the same cumulative share of views. This indicates a more distributed engagement across across what is likely a broader range of content.  The fraction of collaborative videos is nearly five times higher in Music than in Entertainment! This striking difference raises a critical question: do these collaborations provide a tangible advantage in driving audience engagement, or are they merely a characteristic of the category’s content strategy?

The contenders circle each other, eyes locked, each meticulously studying their opponent’s every move. It’s clear they have done their homework. You can almost hear their thoughts as they assess strengths, weaknesses, and opportunities. Here is what they have likely picked up during this intense standoff.

In the Music category, collaborative videos reveal a clear advantage. The data show that collaborations lead to higher averages in views, likes, and even dislikes by factors of 1.74, 1.68 and 1.60 respectively.</div>""", unsafe_allow_html=True)

    music_collab_vs_non_collab_fig = os.path.join(FIGURES_PDF, "Music_Collab_vs_NonCollab__Hist_Boxplot.pdf")
    pdf_viewer(music_collab_vs_non_collab_fig, width=900, height=450)

    st.markdown("""<div style="text-align: justify;">
    The shifts in distribution are quite clear, with collaborative videos pulling ahead. However, this rise in engagement is not without its nuances: the increase in dislikes suggests that while collaborations draw more attention, they may also provoke more polarized reactions. In our statistical analysis we always observe p-values below many orders of magnitudes bellow the significance threshold $0.05$. This confirms that the distributions between collaborative and non-collaborative videos in Music are distinctly different, solidifying the collaboration effect as a driver of engagement.

In Entertainment, the impact of collaboration is present but less pronounced. Collaborations result in a similar increase for likes (factor 1.65) a more moderate one for views (by factors 1.25), but the difference between collaborative and non-collaborative distributions is narrower than in Music.
    </div>""", unsafe_allow_html=True)

    file2 = os.path.join(FIGURES_PDF, "Entertainment_Collab_vs_NonCollab__Hist_Boxplot.pdf")
    pdf_viewer(file2, width=900, height=450)

    st.markdown("""
                <div style="text-align: justify;">
    Interestingly, the average number of dislikes remains nearly identical between the two groups. This suggests that while collaborations in Entertainment bring noticeable gains in positive engagement, they do not introduce the same level of polarizing reactions observed in Music. Here too, statistical tests confirm that collaborations add value, though less significantly (in terms of total engagement) than in the Music category.

    As the match intensifies, we turn our attention to the channels themselves—the fighters behind the punches. The collaboration ratios, defined as the number of collaborative videos divided by the total number of videos, reveal a strong difference in strategy between the two contenders. Music channels, like seasoned boxers with a well-honed jab, show a greater tendency to collaborate, striking more frequently with partnerships.
    </div>""", unsafe_allow_html=True)

    file3 = os.path.join(FIGURES_PDF, "top_0.95_views__Hist_Boxplot.pdf")
    pdf_viewer(file3, width=900, height=450)

    st.markdown("""
                <div style="text-align: justify;">
    In contrast, Entertainment channels adopt a more reserved approach, with fewer collaborations overall. Their collaboration ratios are more clustered (see the boxplot), indicating a consistent but limited approach to partnerships. Meanwhile, Music channels show a wider spread, reflecting a more dynamic and varied strategy, with some channels heavily relying on collaborations to dominate the field. This broader distribution highlights Music's willingness to experiment and use partnerships as a key tactic in the fight for audience engagement. 

    As the fight progresses, the contenders reveal yet another layer of their strategies, this time through the channels contributing to their audience bases—the heavyweights and undercards of their respective corners.
    When focusing on channels that account for 60% of the total subscribers in each category, we see distinct approaches reminiscent of a seasoned boxer’s game plan.
    </div>""", unsafe_allow_html=True)


    file4 = os.path.join(FIGURES_PDF, "top_0.6_channels__Hist_Boxplot.pdf")
    pdf_viewer(file4, width=900, height=450)


    st.markdown("""
                <div style="text-align: justify;">
    For Music channels, the top performers fight like disciplined champions, with a clear and refined strategy. The histogram reveals an "empty" range between 0.8 and 0.9875 for collaboration ratios among the top channels, as if these fighters avoid unnecessary punches, relying instead on precise and impactful moves.  The boxplot shows that the upper percentiles and median are higher for the top channels. This reflects a deliberate strategy by the Music heavyweights, using collaborations as their winning combination to dominate the ring.

    Entertainment channels on the other hand, resemble a more unpredictable fighter, with gaps in their collaboration ratio distribution indicating an inconsistent approach. The medians and means for top and bottom channels are nearly identical, suggesting that both the heavyweights and undercards employ similar strategies. Despite some success, the Entertainment category appears less focused on collaborations as a decisive punch, lacking the clear differentiation seen in Music. 

    The Verdict: Music Wins the Round 10-9.

    Music edges out Entertainment in this round, earning a narrow victory in this round. Its strategic use of collaborations as a tool for dominance gives it the upper hand, despite the polarized reactions it sometimes generates

    As the fighters return to their corners, the coaches are already strategizing for the rounds ahead. After two rounds focusing on monetization potential and collaboration, a new question emerges: if the fight were to last, who would have the most endurance?
    </div>""", unsafe_allow_html=True)


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

    """Here we are, the final round of the greatest showdown of the century! As the tension peaks, the two contenders size each other up, daring, preparing for the ultimate clash. In one corner of the ring stands Entertainment, the unrivaled master of seasons, holidays, and global events. In the other, Music, backed by its loyal audience, blockbuster album releases, and powerful collaborations. Who will emerge victorious from this epic battle? Who will claim the coveted title of YouTube champion? The bets are placed, the predictions are in. One thing is certain: tonight, the story of the world's most popular video platform will be rewritten."""

    """What better theme for the final showdown than exploring the diversity of content? Stepping out of one’s comfort zone is never easy, but the future often belongs to the bold. Before we dive into measuring diversity itself, let’s first take a closer look at the most common themes within our two heavyweight categories: Music and Entertainment."""

    """To uncover the key domains of expertise for tonight's contenders, we dive into the titles and tags of their respective videos. That's where TF-IDF comes in, a somewhat intimidating name for a simple yet powerful method. It evaluates a word's importance in a text by considering how often it appears in that text and how rare it is across all texts. Using this approach, we pinpoint the most significant words. 
    The words from titles and tags are then transformed into numerical values, enabling us to leverage Machine Learning algorithms for clustering. Clustering organizes similar words into distinct groups. From there, we can associate each video with the group of similar words that best defines it, and identify the most significant word in the group as the video's primary theme."""

    """When we visualize the themes of each category as word clouds, where the size of each word reflects its prominence, we get this delightful illustration:"""

    # Overall time evolution of delta_views
    fig1_path = os.path.join(FIGURES_PKL, "wordcloud_music_vs_entertainment_general.pkl")
    with open(fig1_path, "rb") as f:
        fig1 = pickle.load(f)
    svg_buffer = io.StringIO()
    fig1.savefig(svg_buffer, format='svg', bbox_inches='tight')
    svg_buffer.seek(0)
    svg_data = svg_buffer.getvalue()
    svg_data_scaled = svg_data.replace('<svg ', '<svg style="width:100%; height:100%;" ')
    st.components.v1.html(f"<div>{svg_data_scaled}</div>", height=300, scrolling=True)

    """But now, it’s time to quantify this diversity and crown the evening's ultimate champion—the suspense is unbearable! Now that we have identified the main themes for each category, we can calculate the average distance between them. The greater this distance, the more diverse the themes; the smaller it is, the less varied they are. 

But here’s the burning question: how on earth do you calculate the distance between words? The answer is surprisingly straightforward! We map the words into a high-dimensional space using a pre-trained embedding model. Simple, right? 

In essence, each word is assigned a vector, and we calculate the average distance between these vectors. This average distance becomes our measure of diversity—voilà! When we visualize the distribution of diversity scores for the top 10,000 videos in each category, this is what it looks like:"""

    # Overall time evolution of delta_views
    fig1_path = os.path.join(FIGURES_PKL, "diversity_histogram.pkl")
    with open(fig1_path, "rb") as f:
        fig1 = pickle.load(f)
    svg_buffer = io.StringIO()
    fig1.savefig(svg_buffer, format='svg', bbox_inches='tight')
    svg_buffer.seek(0)
    svg_data = svg_buffer.getvalue()
    svg_data_scaled = svg_data.replace('<svg ', '<svg style="width:100%; height:100%;" ')
    st.components.v1.html(f"<div>{svg_data_scaled}</div>", height=600, scrolling=True)


    """We notice that the distributions are quite similar, but Music shows a slightly higher average diversity. Yet, something feels off—we're not accounting for the timeline of the videos, i.e., how diversity evolves over time and which category ultimately crosses the finish line as the most diverse.""" 

    # Overall time evolution of delta_views
    fig1_path = os.path.join(FIGURES_PKL, "diversity_music_vs_entertainment_days.pkl")
    with open(fig1_path, "rb") as f:
        fig1 = pickle.load(f)
    svg_buffer = io.StringIO()
    fig1.savefig(svg_buffer, format='svg', bbox_inches='tight')
    svg_buffer.seek(0)
    svg_data = svg_buffer.getvalue()
    svg_data_scaled = svg_data.replace('<svg ', '<svg style="width:100%; height:100%;" ')
    st.components.v1.html(f"<div>{svg_data_scaled}</div>", height=800, scrolling=True)


    """When we examine diversity with fine granularity, day by day, we observe significant variance at the start for both categories, which gradually stabilizes over time. But what if we take a step back and adopt a broader perspective for a clearer picture?"""

    # Overall time evolution of delta_views
    fig1_path = os.path.join(FIGURES_PKL, "diversity_music_vs_entertainment_months.pkl")
    with open(fig1_path, "rb") as f:
        fig1 = pickle.load(f)
    svg_buffer = io.StringIO()
    fig1.savefig(svg_buffer, format='svg', bbox_inches='tight')
    svg_buffer.seek(0)
    svg_data = svg_buffer.getvalue()
    svg_data_scaled = svg_data.replace('<svg ', '<svg style="width:100%; height:100%;" ')
    st.components.v1.html(f"<div>{svg_data_scaled}</div>", height=800, scrolling=True)


    """Things are becoming clearer now—we see that both diversity curves follow an upward trend over time. This indicates that as time goes on, the content in both Music and Entertainment becomes increasingly varied. Fascinating insights! 

However, it's still tricky to pinpoint the ultimate winner, as the two curves remain close to one another. Let’s step even further back and use years as our time window for a broader perspective."""

    # Overall time evolution of delta_views
    fig1_path = os.path.join(FIGURES_PKL, "diversity_music_vs_entertainment_year.pkl")
    with open(fig1_path, "rb") as f:
        fig1 = pickle.load(f)
    svg_buffer = io.StringIO()
    fig1.savefig(svg_buffer, format='svg', bbox_inches='tight')
    svg_buffer.seek(0)
    svg_data = svg_buffer.getvalue()
    svg_data_scaled = svg_data.replace('<svg ', '<svg style="width:100%; height:100%;" ')
    st.components.v1.html(f"<div>{svg_data_scaled}</div>", height=800, scrolling=True)

    # Overall time evolution of delta_views
    fig1_path = os.path.join(FIGURES_PKL, "diversity_music_vs_entertainment_years_zoom.pkl")
    with open(fig1_path, "rb") as f:
        fig1 = pickle.load(f)
    svg_buffer = io.StringIO()
    fig1.savefig(svg_buffer, format='svg', bbox_inches='tight')
    svg_buffer.seek(0)
    svg_data = svg_buffer.getvalue()
    svg_data_scaled = svg_data.replace('<svg ', '<svg style="width:100%; height:100%;" ')
    st.components.v1.html(f"<div>{svg_data_scaled}</div>", height=800, scrolling=True)

    """And there it is—crystal clear! We can now see how the diversity scores of the two categories intertwined over time. At first, Entertainment held the lead, only to be overtaken by Music, which gained a comfortable advantage. But in a stunning turn of events, Entertainment, against all odds, made an incredible comeback, snatching the lead and finishing as the winner! 

Unbelievable but true: Entertainment claims the diversity crown in this round, by a narrow margin! What an extraordinary finish!"""

        

elif selected_section == "Conclusion":
    st.header("Conclusion")

    """After four intense rounds, the judges have tallied their scores, and the verdict is in. Here’s how the fight unfolded:

**Round 1:** Monetization Potential – Entertainment narrowly edged out Music with a 10-9 victory, thanks to its flexibility in ad placement and longer video durations.
    
**Round 2:** Weekly Popularity & Consistency – Music came back strong, proving its spikes in views are longer-lasting and decay more slowly, winning this round decisively 10-8.

**Round 3:** Collaborations – Music showcased its ability to drive engagement through partnerships, delivering a solid 10-9 win.

**Round 4:** Diversity – Entertainment claimed the final round in a nail-biter, pulling ahead in thematic variety over time with a 10-9 score.

#### **Final Score: Music 38 – Entertainment 37** 🥊 

By the narrowest of margins, Music emerges victorious, clinching the title in a spectacularly close match. It was a fight for the ages, with both contenders showcasing their unique strengths and strategies. Music’s strong collaborations and enduring popularity ultimately secured its triumph, while Entertainment’s adaptability and creative range made it a formidable opponent to the very end.

But as the fighters leave the ring, there’s a lingering feeling that the story isn’t over. Certain angles—perhaps critical ones—were left unexplored. Could Entertainment’s breadth of content appeal have been measured differently? Could Music’s replayability reveal even deeper insights? The questions hang in the air, leaving fans and analysts alike yearning for a rematch to settle the score definitively.

The crowd roars, the fighters shake hands, and the ring lights dim. What a battle! Until the next showdown, keep watching, keep analyzing, and never stop exploring the world of YouTube."""


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
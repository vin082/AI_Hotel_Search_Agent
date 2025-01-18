import streamlit as st
from crewai import Agent, Task, Process, Crew
from crewai_tools import SerperDevTool, WebsiteSearchTool
import os
from dotenv import load_dotenv
from datetime import datetime, date
import pandas as pd

def load_environment():
    load_dotenv()
    return os.getenv('OPENAI_API_KEY'), os.getenv('SERPER_API_KEY')

def calculate_days(check_in_date, check_out_date):
    # Calculate the difference between dates
    delta = check_out_date - check_in_date
    return delta.days

def create_agents(destination, days, budget, people, check_in_date, check_out_date):
    search_tool = SerperDevTool()
    web_rag_tool = WebsiteSearchTool()
    
    researcher = Agent(
        role='Travel Planner Expert',
        goal=f'Scrape the web to find the best deals for {destination}, stay={days} days, budget=approx {budget} USD,{people},{check_in_date} and {check_out_date}from booking.com website.',
        backstory='An expert analyst in planning travel',
        tools=[search_tool, web_rag_tool],
        verbose=True 
    )
    
    planner = Agent(
        role='Travel Planner',
        goal='Generate customized itineraries of hotels based on user preferences',
        backstory='An expert in generating hotel itinerary',
        verbose=True
    )
    
    return researcher, planner

def create_tasks(researcher, planner):
    research = Task(
        description="Find the best itinerary for a user based on their preferences",
        expected_output='A summary hotels itineraries including Hotel Name, City, Cost per day, Amenities',
        agent=researcher
    )
    
    planning = Task(
        description="Generate a customized itinerary for a user based on their preferences",
        expected_output='Shortlist top 10 hotel deals for the user based on their preferences. Include the booking URLs in the response',
        agent=planner
    )
    
    return research, planning

def run_travel_planner(destination, days, budget, people, check_in_date, check_out_date):
    # Create agents and tasks
    researcher, planner = create_agents(destination, days, budget, people, check_in_date, check_out_date)
    research_task, planning_task = create_tasks(researcher, planner)
    
    # Create and run the crew
    crew = Crew(
        agents=[researcher, planner],
        tasks=[research_task, planning_task],
        process=Process.sequential,
        verbose=True
    )
    
    return crew.kickoff()

def main():
    # Initialize session state for storing results
    if 'itinerary_result' not in st.session_state:
        st.session_state.itinerary_result = None
    
    # Page configuration
    st.set_page_config(page_title="AI Travel Planner ✈️", layout="wide")
    
    # Header
    st.title("AI Hotel Planner 🏰")
    st.caption("Plan your next adventure with AI Travel Planner by researching and planning a personalized itinerary")
    
    # Create columns for inputs
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        destination = st.text_input("Destination", placeholder="Enter city name (e.g., Mumbai)")
    
    with col2:
        people = st.number_input("Number of people", min_value=1, value=1)
    
    with col3:
        budget = st.number_input("Budget (USD)", min_value=100, max_value=50000, value=1000)
    
    with col4:
        today = date.today()
        check_in_date = st.date_input("Check-in Date", 
                                     value=today,
                                     min_value=today,
                                     format="YYYY/MM/DD")
    
    with col5:
        # Set minimum check-out date to be the day after check-in
        min_checkout = check_in_date + pd.Timedelta(days=1)
        check_out_date = st.date_input("Check-out Date",
                                      value=min_checkout,
                                      min_value=min_checkout,
                                      format="YYYY/MM/DD")
    
    # Calculate and display the number of days
    days = calculate_days(check_in_date, check_out_date)
    st.info(f"Duration of stay: {days} days")

    # Generate Itinerary button
    if st.button("Generate Itinerary", type="primary"):
        if not destination:
            st.error("Please enter a destination")
            return
        
        if days < 1:
            st.error("Check-out date must be after check-in date")
            return
        
        try:
            # Show loading spinner while generating itinerary
            with st.spinner('Generating your personalized travel itinerary...'):
                # Load API keys
                openai_api_key, serp_api_key = load_environment()
                
                if not openai_api_key or not serp_api_key:
                    st.error("Missing API keys. Please check your environment variables.")
                    return
                
                # Run the travel planner and store result in session state
                result = run_travel_planner(destination, days, budget, people, check_in_date, check_out_date)
                st.session_state.itinerary_result = str(result)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please try again or contact support if the problem persists.")

    # Display results and download button if we have results
    if st.session_state.itinerary_result:
        # Display results in an expander
        with st.expander("Your Personalized Travel Itinerary", expanded=True):
            st.markdown(st.session_state.itinerary_result)
        
        # Add download button for the itinerary
        st.download_button(
            label="Download Itinerary",
            data=st.session_state.itinerary_result,
            file_name=f"travel_itinerary_{destination}.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()
import pandas as pd
import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from soccerplots.radar_chart import Radar
from scipy.stats import percentileofscore
from datetime import datetime


st.set_page_config(page_title = 'Datascoutingstool Kaa Gent',
                   page_icon = ':bar_chart:',
                   layout="wide")

st.markdown("""
    <style>
        .title-wrapper {
            display: flex;
            align-items: center;
        }
        .icon {
            margin-right: 10px;
        }
    </style>
""", unsafe_allow_html=True)

def title_with_icon(icon, title):
    st.markdown(f"<div class='title-wrapper'><div class='icon'>{icon}</div><h4>{title}</h4></div>", unsafe_allow_html=True)

def FysiekeDashboard():
    #functie om de percentielscores ve dataframe te berekenen (alle kolommen die geen percentiel moeten hebben in de index plaatsen)
    def bereken_percentiel_score(B):
        percentiel_scores_dict = {}
        for kolomnaam in B.columns:
            percentiel_scores_dict[kolomnaam] = B[kolomnaam].apply(lambda x: percentileofscore(B[kolomnaam], x))
        return pd.DataFrame(percentiel_scores_dict)
    # geboortedatum omzetten naar leeftijd
    def bereken_leeftijd(geboortedatum):
        leeftijd = huidige_datum.year - geboortedatum.year - ((huidige_datum.month, huidige_datum.day) < (geboortedatum.month, geboortedatum.day))
        return leeftijd

    #Inlezen bestanden met de fysieke data van alle competities en samenvoegen
    C1 = pd.read_csv(r"D1.csv", encoding='utf-8', sep=';')
    C2 = pd.read_csv(r"D2.csv", encoding='utf-8', sep=';')
    C3 = pd.read_csv(r"D3.csv", encoding='utf-8', sep=';')
    C4 = pd.read_csv(r"D4.csv", encoding='utf-8', sep=';')
    C5 = pd.read_csv(r"D5.csv", encoding='utf-8', sep=';')
    C6 = pd.read_csv(r"D6.csv", encoding='utf-8', sep=';')
    Big_df = pd.concat([C1, C2, C3, C4, C5, C6])
    #Goedzetten dataframe
    Big_df.drop(columns= 'Match', inplace=True)
    Big_df['Birthdate'] = pd.to_datetime(Big_df['Birthdate'])
    huidige_datum = datetime.now()
    Big_df['Age'] = Big_df['Birthdate'].apply(bereken_leeftijd)
    Big_df.drop('Birthdate', axis=1, inplace=True)
    df_allplayers = Big_df.groupby(['Player', 'Position', 'Team', 'Competition', 'Position Group', 'Age']).sum()
    df_psv99 = Big_df.groupby(['Player', 'Position', 'Team', 'Competition', 'Position Group', 'Age']).mean()  
    df_allplayesrP90 = (df_allplayers.iloc[:,:].div(df_allplayers['Minutes Played'], axis=0) * 90)
    df_allplayesrP90['Minutes Played'] = df_allplayers['Minutes Played']
    df_allplayesrP90['Topspeed'] = df_psv99['PSV-99']
    df_allplayesrP90['Matches Played'] = df_allplayesrP90['Minutes Played']/ 90
    df_allplayesrP90['Matches Played'] = df_allplayesrP90['Matches Played'].round(0)
    df_allplayesrP90 = df_allplayesrP90.drop(df_allplayesrP90.loc[df_allplayesrP90['Matches Played'] < 3].index)
    #df_allplayesrP90.set_index('Player')
    A = df_allplayesrP90.reset_index()
    A = A.set_index('Player')
    A['Minutes Played'] = A['Matches Played']
    A.drop(columns= 'Matches Played', inplace= True)
    A.drop(columns= 'PSV-99', inplace= True)
    A = A.rename(columns={'Minutes Played': 'Matches Played'})
    A = A.round(1)


 




    #filters implementeren
    st.subheader('Fysieke Scoutingstool')
    st.markdown("")
    st.markdown("Beschrijving:")
    st.markdown("- In dit dashboard kan de fysieke data gevonden worden van 20+ competities, afkomstig uit skillcorner.")
    st.markdown("- Filter op basis van verschillende criteria om een subset te krijgen van de dataset.")
    st.markdown("- Sorteer de dataset op basis van bepaalde parameters.")
    st.markdown("- Van de gefilterde dataset worden ook de percentielscores weergegeven (Zie tabel 2).")
    st.markdown("- De bedoeling is om snel zicht te krijgen op de fysiek sterkste spelers binnen alle competities.")
    st.divider()
    st.markdown("")
    col1, col2, col3, col4, col5 = st.columns([3, 0.75, 3, 0.75, 2])
    with col1:
        positiegroep = st.multiselect('positiegroep', A['Position Group'].unique())
        if positiegroep:
            A = A.loc[A['Position Group'].isin(positiegroep)]
        positie = st.multiselect('positie', A['Position'].unique())
        if positie:
            A = A.loc[A['Position'].isin(positie)]
        # Bepaal de minimale en maximale leeftijd in het DataFrame
        
        competitie = st.multiselect('competitie', A['Competition'].unique())
        if competitie:
            A = A.loc[A['Competition'].isin(competitie)]
        club = st.multiselect('team', A['Team'].unique())
        if club:
            A = A.loc[A['Team'].isin(club)]
    with col3:
        min_age = int(A['Age'].min())
        max_age = int(A['Age'].max())
        leeftijd = st.slider('Leeftijd', min_value=17, max_value=42, value=(17, 42))
        A = A[(A['Age'] >= leeftijd[0]) & (A['Age'] <= leeftijd[1])]
        min_matches = int(A['Matches Played'].min())
        max_matches = int(A['Matches Played'].max())
        matchen = st.slider('Matchen', min_value=3, max_value=40, value=(3, 40))
        A = A[(A['Matches Played'] >= matchen[0]) & (A['Matches Played'] <= matchen[1])]
        metrics = st.multiselect('show parameters', ['Distance P90', 'Running Distance P90', 'HSR Distance P90', 'Sprinting Distance P90', 'HI Distance P90', 'Count HSR P90', 'Count Sprint P90', 'Count HI P90', 'Count Medium Acceleration P90', 'Count High Acceleration P90', 'Count Medium Deceleration P90', 'Count High Deceleration P90', 'Topspeed'])
        general = st.multiselect('show info', ['Position', 'Position Group', 'Age', 'Team', 'Competition', 'Matches Played'])
        if not general:
            general = ['Position', 'Position Group', 'Age', 'Team', 'Competition', 'Matches Played']
        if not metrics:
            metrics =  ['Distance P90', 'Running Distance P90', 'HSR Distance P90', 'Sprinting Distance P90', 'HI Distance P90', 'Count HSR P90', 'Count Sprint P90', 'Count HI P90', 'Count Medium Acceleration P90', 'Count High Acceleration P90', 'Count Medium Deceleration P90', 'Count High Deceleration P90', 'Topspeed']

    with col5:
        filter = st.multiselect('filter parameters', ['Distance P90', 'Running Distance P90', 'HSR Distance P90', 'Sprinting Distance P90', 'HI Distance P90', 'Count HSR P90', 'Count Sprint P90', 'Count HI P90', 'Count Medium Acceleration P90', 'Count High Acceleration P90', 'Count Medium Deceleration P90', 'Count High Deceleration P90', 'Topspeed'])
        if filter:
            for parameter in filter:
                min_val = int(A[parameter].min())
                max_val = int(A[parameter].max())
                parafilter = st.slider(parameter, min_value=min_val, max_value=max_val, value=(min_val, max_val))
                A = A[(A[parameter] >= parafilter[0]) & (A[parameter] <= parafilter[1])]



    weergave = general + metrics 
    A2 = A[weergave]
    st.markdown("")
    st.dataframe(A2, height = 700)



    #st.dataframe(A.describe().round(1))
    
    B = A.reset_index()
    B = B.set_index(['Player', 'Position', 'Position Group', 'Age', 'Team', 'Competition', 'Matches Played'])
    #st.dataframe(B) 
    
    st.markdown("##### Percentielscores van de subset")
    st.markdown("")
    st.markdown("Extra parameters:")
    st.markdown("- **Explosivity:** parameter waarin zowel de topsnelheid, het aantal acceleraties, het aantal sprints en de totale sprintafstand samen wordt genomen.")
    st.markdown("- **Volume:** parameter waarin de totale loopafstand, high intensity afstand alsook het aantal high intensity runs samen wordt opgenomen.")
    st.markdown("- **Total:** Het gemiddelde van Explosiviteit en Volume. Een algemene maatstaf om zowel de snelheid als het loopvermogen van een speler in kaart te brengen.")
    st.divider()
    C = bereken_percentiel_score(B)
    C['Explosivity'] = (((C['Topspeed'] * 2)+ (C['Count High Acceleration P90'] * 2) + (C['Count High Deceleration P90']*0.75) + (C['Count Sprint P90'] * 0.75) + (C['Sprinting Distance P90'] * 0.5))/6)
    C['Volume'] = (((C['Distance P90'] * 2) + (C['Running Distance P90']*0.75) + (C['HI Distance P90']*0.75) + (C['Count HI P90']*0.75))/4.5)
    C['Total'] = ((C['Volume']+ C['Explosivity'])/2)

    col1, col2, col3, col4, col5, col6, col7 = st.columns([3, 0.5, 3, 0.5, 3, 0.5, 3])
    with col1: 
        parameterslider = ['Distance P90', 'Running Distance P90', 'HSR Distance P90', 'Sprinting Distance P90']
        for parameter in parameterslider:
            percentielfilter = st.slider(parameter, min_value=0, max_value=100, value=(0, 100))
            C = C[(C[parameter] >= percentielfilter[0]) & (C[parameter] <= percentielfilter[1])]
    with col3: 
        parameterslider = ['HI Distance P90', 'Count HSR P90', 'Count Sprint P90', 'Count HI P90']
        for parameter in parameterslider:
            percentielfilter = st.slider(parameter, min_value=0, max_value=100, value=(0, 100))
            C = C[(C[parameter] >= percentielfilter[0]) & (C[parameter] <= percentielfilter[1])]
    with col5: 
        parameterslider = ['Count Medium Acceleration P90', 'Count High Acceleration P90', 'Count Medium Deceleration P90', 'Count High Deceleration P90']
        for parameter in parameterslider:
            percentielfilter = st.slider(parameter, min_value=0, max_value=100, value=(0, 100))
            C = C[(C[parameter] >= percentielfilter[0]) & (C[parameter] <= percentielfilter[1])]
    with col7: 
        parameterslider = ['Topspeed', 'Volume', 'Explosivity', 'Total']
        for parameter in parameterslider:
            percentielfilter = st.slider(parameter, min_value=0, max_value=100, value=(0, 100))
            C = C[(C[parameter] >= percentielfilter[0]) & (C[parameter] <= percentielfilter[1])]




    C = C.reset_index()
    C = C.set_index('Player')
    drrest = ['Volume', 'Explosivity', 'Total']
    weergave2 = weergave + drrest
    C2 = C[weergave2]
    st.dataframe(C2.round(1), height = 700)







def radarcharts():
    list_attackers = ['Oh Hyeon-Gyu', 'T. Tissoudali', 'L. Depoitre', 'G. Orban', 'H. Cuypers', 'A. Lind', 'A. Gudjohnsen', 'M. Breunig', 'M. Biereth', 'G. Nilsson', 'K. Dolberg', 'T. Arokodare', 'I. Thiago', 'K. Denkey', 'A. Ementa', 'D. Ayensa', 'V. Janssen', 'M. Amoura', 'F. Mayulu', 'R. Durosinmi', 'M. Dean', 'C. Shpendi', 'A. Hountondji', 'F. Abiuso', 'G. Borrelli', 'W. Geubbels', 'A. Peralta'] #'A. Sieb', 'Rwan Cruz', 'Ludo Thiago', 'T. Barry', 'I. Matanović', 'M. Biereth', 'A. Gudjohnsen''M. Fernandez-Pardo''Cho_Gue-Sung', 'N. Milanovic', 'D. Maldini','A. Crnac''P. Ratkov', 'F. Girotti', 'Y. Salech', 'R. Čaks','B. Barišić', 'B. Nsimba', ]
    list_centralmidfield = ['S. Kums', 'J. De_Sart', 'O. Gandelman', 'P. Gerkens', 'R. Onyedika', 'C. Nielsen', 'H. Vetlesen', 'P. Hrošovský', 'B. Heynen', 'M. Galarza', 'L. Lopes', 'H. Van_der_Bruggen', 'A. Vermeeren', 'M. Keita', 'A. Yusuf', 'T. Leoni', 'M. Rits', 'N. Sadiki', 'M. Rasmussen', 'C. Vanhoutte', 'L. Amani', 'A. Kadri',  'O. Højlund', 'P. Berg', 'P. Aaronson', 'O. Sørensen', 'T. Rigo', 'A. Bernede', 'L. Bate', 'A. Omgba', 'B. Krushynskyi', 'D. Moses'] #,'S. Esposito',  'A. Omgba_8','J. Pršir', 'N. Souren', 'A. Omgba_10', 'M. Kusu','F. Krastev', 'F. Krastev_total', 'C. Rodri', 'A. Bernede', 'L. Bate', 'F. Krastev', 'F. Krastev_total''C. Rodri', , 'S. Resink','S. Esposito', 'J. Bakke', 'R. Pukštas', 'A. Morris',  'G. Busio',
    list_attackingmidfield = ['H. Hong', 'A. Hjulsager', 'B. El_Khannouss', 'H. Vanaken', 'J. Ekkelenkamp', 'C. Puertas', 'A. Omgba', 'A. Karabec']
    list_wingers = ['D. Yokota', 'M. Fofana', 'M. Sonko', 'A. Sanches', 'J. Steuckers', 'T. Hazard', 'A. Nusa',  'A. Dreyer', 'C. Ejuke', 'M. Balikwisha', 'A. Minda', 'T. Somers', 'J. Paintsil', 'A. Skov_Olsen', 'P. Zinckernagel']
    list_wingbacks = [ 'M. Samoise', 'A. Brown', 'N. Fadiga', 'L. Lapoussin', 'A. Castro-Montes', 'M. De_Cuyper', 'B. Meijer', 'Z. El_Ouahdi', 'J. Kayembe', 'G. Arteaga', 'D. Muñoz', 'H. Siquet', 'O. Wijndal', 'J. Bataille', 'K. Sardella', 'L. Augustinsson']#, 'I. Camara',
    list_centerbacks = ['S. Mitrović', 'J. Torunarigha', 'I. Kandouss',  'T. Watanabe', 'A. Bright', 'N. Abbey', 'Z. Debast', 'T. Alderweireld', 'J. Vertonghen', 'B. Mechele', 'B. Popović', 'C. Burgess', 'C. Cuesta', 'J. Daland', 'J. Spileers', 'K. Machida', 'M. McKenzie', 'R. Sykes', 'Z. Van_den_Bosch', 'A. N\'Diaye', 'T. Cissokho', 'H. Petrov', 'M. Nadé', 'S. Karič', 'A. Tsoungui', 'A. Batagov', 'J. Romsaas', 'N. Ngoy']#, 'José_Marsà','J. Cordoba', 'L. Lochoshvili', 'D. Cornelius','U. Bilbao', , 'P. Awuku', 'E. Cobbaut', 'A. Filin', 'J. Rasmussen', 'S. Kotto', 'S. Kotto2', 'P. Bochniewicz',, 'A. Filin''M. Mbow',, 'E. Cobbaut'

    lijst_general = []
    lijst_attack = []
    lijst_defense = []
    lijst_passing = []
    lijst_fysical = []
    lijst_games = []

    def attacker(df_general, df_defense, df_attack, df_passing, df_fysical, bestandsnaam):
        naam_delen = os.path.splitext(bestandsnaam)[0].split()
        speler_naam = " ".join(naam_delen[2:4])
        general = pd.DataFrame(columns=['Player', 'Total Actions', 'Succesfull Actions', '% Succesfull Actions'])
        general['Player'] = [bestandsnaam] * len(df_general)
        general['Total Actions'] = df_general.iloc[:, 5]
        general['Succesfull Actions'] = df_general.iloc[:, 6]
        general['% Succesfull Actions'] = np.nan
        general_grouped = general.groupby('Player').mean()
        mean_minutes = df_general['Minutes played'].mean()
        general_grouped = ((general_grouped / mean_minutes) * 90).round(2)
        general_grouped['% Succesfull Actions'] = ((general_grouped['Succesfull Actions'] / general_grouped['Total Actions'])*100).round(2)
        games = pd.DataFrame()
        games['Player'] = [bestandsnaam] * 1
        games['Total Games'] = len(general)
        games['Average Minutes per Game'] = mean_minutes.round(2)
        lijst_games.append(games)
        attack = pd.DataFrame()
        attack['Player'] = ([bestandsnaam] * len(df_attack))
        attack['Goals'] = df_attack.iloc[:, 5]
        attack['xG'] = df_attack.iloc[:, 9]
        attack['Assists'] = df_attack.iloc[:, 6]
        attack['xA'] = df_passing.iloc[:, 14]
        attack['Shots'] = df_attack.iloc[:, 7]
        attack['Shots On Target'] = df_attack.iloc[:, 8]
        attack['% Shots On Target'] = np.nan 
        attack['Dribbles'] = df_attack.iloc[:, 13]
        attack['Succesfull Dribbles'] = df_attack.iloc[:, 14]
        attack['% Succesfull Dribbles'] = np.nan
        attack['Offensive Duels'] = df_attack.iloc[:, 15]
        attack['Won Offensive Duels'] = df_attack.iloc[:, 16]
        attack['% Won Offensive Duels'] = np.nan
        attack['Touches in Penalty Area'] = df_attack.iloc[:, 17]
        attack['Offsides'] = df_attack.iloc[:, 18]
        attack['Progressive Runs'] = df_attack.iloc[:, 19]
        attack['Fouls Suffered'] = df_attack.iloc[:, 20]
        attack_grouped = attack.groupby('Player').mean()
        attack_grouped = ((attack_grouped / mean_minutes) * 90).round(2)
        attack_grouped['% Shots On Target'] = ((attack_grouped['Shots On Target'] / attack_grouped['Shots']) * 100).round(1)
        attack_grouped['% Succesfull Dribbles'] = ((attack_grouped['Succesfull Dribbles'] / attack_grouped['Dribbles']) * 100).round(1)
        attack_grouped['% Won Offensive Duels'] = ((attack_grouped['Won Offensive Duels'] / attack_grouped['Offensive Duels']) * 100).round(1)

        passing = pd.DataFrame()
        passing['Player'] = ([bestandsnaam] * len(df_passing))
        passing['Passes'] = df_passing.iloc[:, 5]
        passing['Accurate Passes'] = df_passing.iloc[:, 6]
        passing['% Accurate Passes'] = np.nan
        passing['Long Passes'] = df_passing.iloc[:, 7]
        passing['Accurate Long Passes'] = df_passing.iloc[:, 8]
        passing['% Accurate Long Passes'] = np.nan
        passing['Through Passes'] = df_passing.iloc[:, 9]
        passing['Accurate Through Passes'] = df_passing.iloc[:, 10]
        passing['% Accurate Through Passes'] = np.nan
        passing['Crosses'] = df_passing.iloc[:, 11]
        passing['Accurate Crosses'] = df_passing.iloc[:, 12]
        passing['% Accurate Crosses'] = np.nan
        passing['Passes to Final Third'] = df_passing.iloc[:, 16]
        passing['Accurate Passes to Final Third'] = df_passing.iloc[:, 17]
        passing['% Accurate Passes to Final Third'] = np.nan
        passing['Passes to Penalty Area'] = df_passing.iloc[:, 18]
        passing['Accurate Passes to Penalty Area'] = df_passing.iloc[:, 19]
        passing['% Accurate Passes to Penalty Area'] = np.nan
        passing['Forward Passes'] = df_passing.iloc[:, 21]
        passing['Accurate Forward Passes'] = df_passing.iloc[:, 22]
        passing['% Accurate Forward Passes'] = np.nan
        passing['Back Passes'] = df_passing.iloc[:, 23]
        passing['Accurate Back Passes'] = df_passing.iloc[:, 24]
        passing['% Accurate Back Passes'] = np.nan
        passing_grouped = passing.groupby('Player').mean()
        passing_grouped = ((passing_grouped / mean_minutes) * 90).round(2)
        passing_grouped['% Accurate Passes'] = ((passing_grouped['Accurate Passes'] / passing_grouped['Passes'])*100).round(1)
        passing_grouped['% Accurate Long Passes'] = ((passing_grouped['Accurate Long Passes'] / passing_grouped['Long Passes'])*100).round(1)
        passing_grouped['% Accurate Through Passes'] = ((passing_grouped['Accurate Through Passes'] / passing_grouped['Through Passes'])*100).round(1)
        passing_grouped['% Accurate Crosses'] = ((passing_grouped['Accurate Crosses'] / passing_grouped['Crosses'])*100).round(1)
        passing_grouped['% Accurate Passes to Final Third'] = ((passing_grouped['Accurate Passes to Final Third'] / passing_grouped['Passes to Final Third'])*100).round(1)
        passing_grouped['% Accurate Passes to Penalty Area'] = ((passing_grouped['Accurate Passes to Penalty Area'] / passing_grouped['Passes to Penalty Area'])*100).round(1)
        passing_grouped['% Accurate Forward Passes'] = ((passing_grouped['Accurate Forward Passes'] / passing_grouped['Forward Passes'])*100).round(1)
        passing_grouped['% Accurate Back Passes'] = ((passing_grouped['Accurate Back Passes'] / passing_grouped['Back Passes'])*100).round(1)

        defense = pd.DataFrame()
        defense['Player'] = ([bestandsnaam] * len(df_defense))
        defense['Defensive Duels'] = df_defense.iloc[:, 5]
        defense['Won Defensive Duels'] = df_defense.iloc[:, 6]
        defense['% Won Defensive Duels'] = np.nan
        defense['Aerial Duels'] = df_defense.iloc[:, 7]
        defense['Won Aerial Duels'] = df_defense.iloc[:, 8]
        defense['% Won Aerial Duels'] = np.nan
        defense['Interceptions'] = df_defense.iloc[:, 13]
        defense['Losses'] = df_defense.iloc[:, 14]
        defense['Losses Own Half'] = df_defense.iloc[:, 15]
        defense['Recoveries'] = df_defense.iloc[:, 16]
        defense['Recoveries Opp. Half'] = df_defense.iloc[:, 17]
        defense['Yellow Cards'] = df_defense.iloc[:, 20]
        defense['Red Cards'] = df_defense.iloc[:, 21]
        defense_grouped = defense.groupby('Player').mean()
        defense_grouped = ((defense_grouped / mean_minutes) * 90).round(2)
        defense_grouped['% Won Defensive Duels'] = ((defense_grouped['Won Defensive Duels'] / defense_grouped['Defensive Duels'])*100).round(2)
        defense_grouped['% Won Aerial Duels'] = ((defense_grouped['Won Aerial Duels'] / defense_grouped['Aerial Duels'])*100).round(2)

        fysical = pd.DataFrame
        fysical = df_fysical.iloc[:,-13:]
        fysical['Player'] = ([bestandsnaam] * len(df_fysical))
        
        cols = list(fysical.columns)
        #nieuwe_volgorde = cols[2:4] + cols[0:2] + [cols[4]]
        nieuwe_volgorde = [cols[-1]] + cols[:-1] 
        fysical = fysical[nieuwe_volgorde]
        #fysical = df_fysical['Minutes Played', 'Distance P90', 'Running Distance P90', 'HSR Distance P90', 'Sprinting Distance P90', 'HI Distance P90', 'Count HSR P90', 'Count Sprint P90', 'Count HI P90', 'Count Medium Acceleration P90', 'Count High Acceleration P90', 'Count Medium Deceleration P90', 'Count High Deceleration P90', 'PSV-99']
        #st.dataframe(fysical)
        #st.dataframe(df_fysical)
        #st.markdown(fysical.columns)

        #'Minutes Played', 'Distance P90', 'Running Distance P90', 'HSR Distance P90', 'Sprinting Distance P90', 'HI Distance P90', 'Count HSR P90', 'Count Sprint P90', 'Count HI P90', 'Count Medium Acceleration P90', 'Count High Acceleration P90', 'Count Medium Deceleration P90', 'Count High Deceleration P90', 'PSV-99'
        fysical_grouped = fysical.groupby('Player').mean()
        lijst_general.append(general_grouped)
        lijst_attack.append(attack_grouped)
        lijst_defense.append(defense_grouped)
        lijst_passing.append(passing_grouped)
        lijst_fysical.append(fysical_grouped)

    lijst_spelersgroep = []

    def attacker_radar(df_general, df_defense, df_attack, df_passing, df_fysical, bestandsnaam):
        naam_delen = os.path.splitext(bestandsnaam)[0].split()
        speler_naam = " ".join(naam_delen[6:8])
        attack = pd.DataFrame()
        mean_minutes = df_general['Minutes played'].mean()
        attack['Player'] = [bestandsnaam] * len(df_general)
        attack['Total Actions'] = df_general.iloc[:, 5]
        attack['Succesfull Actions'] = df_general.iloc[:, 6]
        #attack['% Succesfull Actions'] = np.nan
        attack['Goals'] = df_attack.iloc[:, 5]
        attack['xG'] = df_attack.iloc[:, 9]
        attack['Goals - xG'] = np.nan
        attack['Shots'] = df_attack.iloc[:, 7]
        attack['Goals per Shot Ratio'] = np.nan
        attack['Assists'] = df_attack.iloc[:, 6]
        attack['Passes'] = df_passing.iloc[:, 5]
        attack['Accurate Passes'] = df_passing.iloc[:, 6]
        attack['% Passes'] = np.nan
        attack['Progressive Runs'] = df_attack.iloc[:, 19]
        
        
        #attack['Shots On Target'] = df_attack.iloc[:, 8]
        #attack['% Shots On Target'] = np.nan 
        attack['Dribbles'] = df_attack.iloc[:, 13]
        attack['Succesfull Dribbles'] = df_attack.iloc[:, 14]
        attack['% Dribbles'] = np.nan

        attack['Offensive Duels'] = df_attack.iloc[:, 15]
        attack['Won Offensive Duels'] = df_attack.iloc[:, 16]
        attack['% Offensive Duels'] = np.nan
        attack['Aerial Duels'] = df_defense.iloc[:, 7]
        attack['Won Aerial Duels'] = df_defense.iloc[:, 8]
        attack['% Aerial Duels'] = np.nan
        attack['Losses'] = df_defense.iloc[:, 14]
        attack['Recoveries'] = df_defense.iloc[:, 16]

        attackfys = pd.DataFrame()
        #attackfys['Player'] = [speler_naam] * len(df_general)
        attackfys = pd.DataFrame()
        attackfys['Topspeed'] = df_fysical['PSV-99']
        attackfys['Accelerations'] = df_fysical['Count High Acceleration P90']
        attackfys['Player'] = [bestandsnaam] * len(attackfys)
        attackfys['Total Distance'] = df_fysical['Distance P90']
        attackfys_group = attackfys.groupby('Player').mean()
        attackfys_group.round(2)
        #if speler_naam == ...:
            #attackfys_group.iloc[:,:] = '0'

        attack_grouped = attack.groupby('Player').mean()
        attack_grouped = ((attack_grouped / mean_minutes) * 90).round(2)
        #attack_grouped['% Succesfull Actions'] = ((attack_grouped['Succesfull Actions'] / attack_grouped['Total Actions']) * 100).round(2)
        #attack_grouped['% Shots On Target'] = ((attack_grouped['Shots On Target'] / attack_grouped['Shots']) * 100).round(2)
        attack_grouped['% Dribbles'] = ((attack_grouped['Succesfull Dribbles'] / attack_grouped['Dribbles']) * 100).round(2)
        attack_grouped['% Offensive Duels'] = ((attack_grouped['Won Offensive Duels'] / attack_grouped['Offensive Duels']) * 100).round(2)
        attack_grouped['% Passes'] = ((attack_grouped['Accurate Passes'] / attack_grouped['Passes']) * 100).round(2)
        attack_grouped['% Aerial Duels'] = ((attack_grouped['Won Aerial Duels'] / attack_grouped['Aerial Duels']) * 100).round(2)
        #attack_grouped['Rec./ Losses Ratio'] = (attack_grouped['Recoveries'] / attack_grouped['Losses']).round(2)
        attack_grouped['Goals per Shot Ratio'] = attack_grouped['Goals'] / attack_grouped['Shots']
        attack_grouped['Goals - xG'] = attack_grouped['Goals'] - attack_grouped['xG']
        attack_grouped.drop(columns=['Total Actions', 'Succesfull Actions', 'Dribbles', 'Passes', 
                                    'Accurate Passes', 'Offensive Duels', 'Won Offensive Duels', 'Aerial Duels', 'Won Aerial Duels', 'Losses', 'Recoveries'],
                            inplace= True) #, 'Shots On Target'
        Attackers = attack_grouped.merge(attackfys_group, how= 'left', on = 'Player')
        lijst_spelersgroep.append(Attackers)

    def Winger_radar(df_general, df_defense, df_attack, df_passing, df_fysical, bestandsnaam):
        naam_delen = os.path.splitext(bestandsnaam)[0].split()
        speler_naam = " ".join(naam_delen[6:8])
        attack = pd.DataFrame()
        mean_minutes = df_general['Minutes played'].mean()
        attack['Player'] = [bestandsnaam] * len(df_general)
        attack['Total Actions'] = df_general.iloc[:, 5]
        attack['Succesfull Actions'] = df_general.iloc[:, 6]
        #attack['% Succesfull Actions'] = np.nan
        attack['Goals'] = df_attack.iloc[:, 5]
        #attack['xG'] = df_attack.iloc[:, 9]

        attack['Assists'] = df_attack.iloc[:, 6]
        
        attack['Shots'] = df_attack.iloc[:, 7]
        attack['Goals per Shot Ratio'] = np.nan
        #attack['Shots On Target'] = df_attack.iloc[:, 8]
        #attack['% Shots On Target'] = np.nan 
        attack['Dribbles'] = df_attack.iloc[:, 14]
        attack['Succesfull Dribbles'] = df_attack.iloc[:, 13]
        attack['% Dribbles'] = np.nan
        attack['Progressive Runs'] = df_attack.iloc[:, 19]
        attack['Passes'] = df_passing.iloc[:, 5]
        attack['Accurate Passes'] = df_passing.iloc[:, 6]
        attack['% Passes'] = np.nan
        attack['Crosses'] = df_passing.iloc[:, 12]
        #attack['Accurate Crosses'] = df_passing.iloc[:, 12]
        #attack['% Crosses'] = np.nan
        attack['Passes Final Third'] = df_passing.iloc[:, 17]
        #attack['Accurate Passes Final Third'] = df_passing.iloc[:, 17]
        #attack['% Passes Final Third'] = np.nan

        attack['Losses'] = df_defense.iloc[:, 14]
        attack['Recoveries'] = df_defense.iloc[:, 16]

        attackfys = pd.DataFrame()
        #attackfys['Player'] = [speler_naam] * len(df_general)
        attackfys = pd.DataFrame()
        attackfys['Topspeed'] = df_fysical['PSV-99']
        attackfys['Player'] = [bestandsnaam] * len(attackfys)
        attackfys['High Accelerations'] = df_fysical['Count High Acceleration P90']
        attackfys['HI Distance'] = df_fysical['HI Distance P90']
        
        attackfys_group = attackfys.groupby('Player').mean()
        attackfys_group.round(2)

        attack_grouped = attack.groupby('Player').mean()
        attack_grouped = ((attack_grouped / mean_minutes) * 90).round(2)
        #attack_grouped['% Succesfull Actions'] = ((attack_grouped['Succesfull Actions'] / attack_grouped['Total Actions'])*100).round(2)
        attack_grouped['Goals per Shot Ratio'] = attack_grouped['Goals'] / attack_grouped['Shots']
        attack_grouped['% Dribbles'] = ((attack_grouped['Dribbles'] / attack_grouped['Succesfull Dribbles'])*100).round(2)
        attack_grouped['% Passes'] = ((attack_grouped['Accurate Passes'] / attack_grouped['Passes'])*100).round(2)
        attack_grouped['Rec./ Losses Ratio'] = (attack_grouped['Recoveries'] / attack_grouped['Losses']).round(2)
        #attack_grouped['% Crosses'] = ((attack_grouped['Accurate Crosses'] / attack_grouped['Crosses'])*100).round(2)
        #attack_grouped['% Passes Final Third'] = ((attack_grouped['Accurate Passes Final Third'] / attack_grouped['Passes Final Third'])*100).round(2)
        attack_grouped.drop(columns=['Total Actions', 'Succesfull Actions', 'Succesfull Dribbles', 'Passes', 
                                    'Accurate Passes', 'Losses', 'Recoveries'],
                            inplace= True)
        Attackers = attack_grouped.merge(attackfys_group, how= 'left', on = 'Player')
        lijst_spelersgroep.append(Attackers)
    def Amidfield_radar(df_general, df_defense, df_attack, df_passing, df_fysical, bestandsnaam):
        naam_delen = os.path.splitext(bestandsnaam)[0].split()
        speler_naam = " ".join(naam_delen[6:8])
        attack = pd.DataFrame()
        mean_minutes = df_general['Minutes played'].mean()
        attack['Player'] = [bestandsnaam] * len(df_general)
        attack['Total Actions'] = df_general.iloc[:, 5]
        attack['Succesfull Actions'] = df_general.iloc[:, 6]
        #attack['% Succesfull Actions'] = np.nan
        attack['Goals'] = df_attack.iloc[:, 5]
        attack['Assists'] = df_attack.iloc[:, 6]  
        attack['Shots'] = df_attack.iloc[:, 7]
        attack['Goals per Shot Ratio'] = np.nan
        attack['TDribbles'] = df_attack.iloc[:, 13]
        attack['Dribbles'] = df_attack.iloc[:, 14]
        attack['% Dribbles'] = np.nan
        #attack['Progressive Runs'] = df_attack.iloc[:, 19]
        attack['Passes'] = df_passing.iloc[:, 5]
        attack['Accurate Passes'] = df_passing.iloc[:, 6]
        attack['% Passes'] = np.nan
        attack['TLong Passes'] = df_passing.iloc[:, 7]
        attack['Long Passes'] = df_passing.iloc[:, 8]
        #attack['% Long Passes'] = np.nan
        attack['TThrough Passes'] = df_passing.iloc[:, 9]
        attack['Through Passes'] = df_passing.iloc[:, 10]
        #attack['% Through Passes'] = np.nan
        attack['TPasses Final Third'] = df_passing.iloc[:, 16]
        attack['Passes Final Third'] = df_passing.iloc[:, 17]
        #attack['% Passes Final Third'] = np.nan
        attack['Duels'] = df_general.iloc[:, 20]
        attack['Won Duels'] = df_general.iloc[:, 21]
        attack['% Duels Won'] = np.nan
        attack['Losses'] = df_defense.iloc[:, 14]
        attack['Recoveries'] = df_defense.iloc[:, 16]

        #attackfys['Player'] = [speler_naam] * len(df_general)
        attackfys = pd.DataFrame()
        attackfys['Topspeed'] = df_fysical['PSV-99']
        attackfys['Player'] = [bestandsnaam] * len(attackfys)
        attackfys['HI Distance'] = df_fysical['HI Distance P90']
        attackfys['Distance'] = df_fysical['Distance P90']
        
        #attackfys['Sprints'] = df_fysical['Count Sprints P90']
        attackfys_group = attackfys.groupby('Player').mean()
        attackfys_group.round(2)

        attack_grouped = attack.groupby('Player').mean()
        attack_grouped = ((attack_grouped / mean_minutes) * 90).round(2)
        #attack_grouped['% Succesfull Actions'] = ((attack_grouped['Succesfull Actions'] / attack_grouped['Total Actions'])*100).round(2)
        attack_grouped['% Dribbles'] = ((attack_grouped['Dribbles'] / attack_grouped['TDribbles'])*100).round(2)
        attack_grouped['% Passes'] = ((attack_grouped['Accurate Passes'] / attack_grouped['Passes'])*100).round(2)
        #attack_grouped['% Long Passes'] = (attack_grouped['Accurate Long Passes'] / attack_grouped['Long Passes']).round(2)
        #attack_grouped['% Through Passes'] = (attack_grouped['Accurate Through Passes'] / attack_grouped['Through Passes']).round(2)
        #attack_grouped['% Passes Final Third'] = (attack_grouped['Accurate Passes Final Third'] / attack_grouped['Passes Final Third']).round(2)
        attack_grouped['Rec./ Losses Ratio'] = (attack_grouped['Recoveries'] / attack_grouped['Losses']).round(2)
        attack_grouped['Goals per Shot Ratio'] = attack_grouped['Goals'] / attack_grouped['Shots']
        attack_grouped['% Duels Won'] = ((attack_grouped['Won Duels'] / attack_grouped['Duels'])*100).round(2)
        attack_grouped.drop(columns=['Succesfull Actions', 'Total Actions', 'TDribbles', 'Passes', 'Accurate Passes', 'TLong Passes', 'TThrough Passes', 'TPasses Final Third', 'Recoveries', 'Losses', 'Won Duels', 'Duels'],
                            inplace= True)
        Attackers = attack_grouped.merge(attackfys_group, how= 'left', on = 'Player')
        lijst_spelersgroep.append(Attackers)

    def Cmidfield_radar(df_general, df_defense, df_attack, df_passing, df_fysical, bestandsnaam):
        naam_delen = os.path.splitext(bestandsnaam)[0].split()
        speler_naam = " ".join(naam_delen[6:8])
        attack = pd.DataFrame()
        mean_minutes = df_general['Minutes played'].mean()
        attack['Player'] = [bestandsnaam] * len(df_general)
        attack['Total Actions'] = df_general.iloc[:, 5]
        attack['Succesfull Actions'] = df_general.iloc[:, 6]
        #attack['% Succesfull Actions'] = np.nan
        attack['Goals'] = df_attack.iloc[:, 5]
        attack['Assists'] = df_attack.iloc[:, 6]  
        #attack['Shots'] = df_attack.iloc[:, 7]
        attack['Progressive Runs'] = df_attack.iloc[:, 19]
        attack['TDribbles'] = df_attack.iloc[:, 13]
        attack['Dribbles'] = df_attack.iloc[:, 14]
        attack['% Dribbles'] = np.nan
        #attack['Progressive Runs'] = df_attack.iloc[:, 19]
        attack['Passes'] = df_passing.iloc[:, 5]
        attack['Accurate Passes'] = df_passing.iloc[:, 6]
        attack['% Passes'] = np.nan
        attack['Long Passes'] = df_passing.iloc[:, 8]
        #attack['Accurate Long Passes'] = df_passing.iloc[:, 8]
        #attack['% Long Passes'] = np.nan
        #attack['Through Passes'] = df_passing.iloc[:, 10]
        #attack['Accurate Through Passes'] = df_passing.iloc[:, 10]
        #attack['% Through Passes'] = np.nan
        attack['Passes Final Third'] = df_passing.iloc[:, 17]
        #attack['Accurate Passes Final Third'] = df_passing.iloc[:, 17]
        #attack['% Passes Final Third'] = np.nan
        attack['Forward Passes'] = df_passing.iloc[:, 22]
        attack['Back Passes'] = df_passing.iloc[:, 24]
        attack['Forward/ Back Pass'] = np.nan
        attack['Interceptions'] = df_defense.iloc[:, 13]
        attack['Losses'] = df_defense.iloc[:, 14]
        attack['Recoveries'] = df_defense.iloc[:, 16]
        attack['Defensive Duels'] = df_defense.iloc[:, 5]
        attack['Defensive Duels Won'] = df_defense.iloc[:, 6]
        attack['% Def. Duels Won'] = np.nan
        attackfys = pd.DataFrame()
        #attackfys['Player'] = [speler_naam] * len(df_general)
        attackfys = pd.DataFrame()
        attackfys['Topspeed'] = df_fysical['PSV-99']
        attackfys['Accelerations'] = df_fysical['Count High Acceleration P90']
        attackfys['Player'] = [bestandsnaam] * len(attackfys)
        attackfys['Distance'] = df_fysical['Distance P90']
        attackfys['HI Distance'] = df_fysical['HI Distance P90']
        attackfys_group = attackfys.groupby('Player').mean()
        attackfys_group.round(2)

        attack_grouped = attack.groupby('Player').mean()
        attack_grouped = ((attack_grouped / mean_minutes) * 90).round(2)
        #attack_grouped['% Succesfull Actions'] = ((attack_grouped['Succesfull Actions'] / attack_grouped['Total Actions'])*100).round(2)
        attack_grouped['% Dribbles'] = ((attack_grouped['Dribbles'] / attack_grouped['TDribbles'])*100).round(2)
        attack_grouped['% Passes'] = ((attack_grouped['Accurate Passes'] / attack_grouped['Passes'])*100).round(2)
        #attack_grouped['% Long Passes'] = ((attack_grouped['Accurate Long Passes'] / attack_grouped['Long Passes'])*100).round(2)
        #attack_grouped['% Through Passes'] = ((attack_grouped['Accurate Through Passes'] / attack_grouped['Through Passes'])*100).round(2)
        #attack_grouped['% Passes Final Third'] = ((attack_grouped['Accurate Passes Final Third'] / attack_grouped['Passes Final Third'])*100).round(2)
        attack_grouped['Rec./ Losses Ratio'] = (attack_grouped['Recoveries'] / attack_grouped['Losses']).round(2)
        attack_grouped['Forward/ Back Pass'] = (attack_grouped['Forward Passes'] / attack_grouped['Back Passes']).round(2)
        attack_grouped['% Def. Duels Won'] = ((attack_grouped['Defensive Duels Won'] / attack_grouped['Defensive Duels'])*100).round(2)
        attack_grouped.drop(columns=['Succesfull Actions', 'Total Actions', 'TDribbles', 'Passes', 'Accurate Passes', 'Recoveries', 'Losses', 'Forward Passes', 'Back Passes', 'Defensive Duels Won', 'Defensive Duels'],
                            inplace= True)
        Attackers = attack_grouped.merge(attackfys_group, how= 'left', on = 'Player')
        lijst_spelersgroep.append(Attackers)

    def wingback_radar(df_general, df_defense, df_attack, df_passing, df_fysical, bestandsnaam):
        naam_delen = os.path.splitext(bestandsnaam)[0].split()
        speler_naam = " ".join(naam_delen[6:8])
        attack = pd.DataFrame()
        mean_minutes = df_general['Minutes played'].mean()
        attack['Player'] = [bestandsnaam] * len(df_general)
        attack['Total Actions'] = df_general.iloc[:, 5]
        attack['Succesfull Actions'] = df_general.iloc[:, 6]
        #attack['% Succesfull Actions'] = np.nan
        attack['Assists'] = df_attack.iloc[:, 6]
        attack['Dribbles'] = df_attack.iloc[:, 13]
        attack['Succesfull Dribbles'] = df_attack.iloc[:, 14]
        attack['% Dribbles'] = np.nan
        attack['Passes'] = df_passing.iloc[:, 5]
        attack['Accurate Passes'] = df_passing.iloc[:, 6]
        attack['% Passes'] = np.nan
        attack['Long Passes'] = df_passing.iloc[:, 8]
        attack['Crosses'] = df_passing.iloc[:, 11]
        attack['Accurate Crosses'] = df_passing.iloc[:, 12]
        attack['% Crosses'] = np.nan
        
        #attack['Accurate Long Passes'] = df_passing.iloc[:, 8]
        #attack['% Long Passes'] = np.nan
        attack['Interceptions'] = df_defense.iloc[:, 13]
        attack['Losses'] = df_defense.iloc[:, 14]
        attack['Recoveries'] = df_defense.iloc[:, 16]
        attack['Defensive Duels'] = df_defense.iloc[:, 5]
        attack['Defensive Duels Won'] = df_defense.iloc[:, 6]
        attack['% Defensive Duels Won'] = np.nan
        attack['Aerial Duels'] = df_defense.iloc[:, 7]
        attack['Won Aerial Duels'] = df_defense.iloc[:, 8]
        attack['% Aerial Duels Won'] = np.nan
        attackfys = pd.DataFrame()
        attackfys['Topspeed'] = df_fysical['PSV-99']
        attackfys['High Accelerations'] = df_fysical['Count High Acceleration P90']
        #attackfys['Sprints'] = df_fysical['Count Sprint P90']
        attackfys['Player'] = [bestandsnaam] * len(attackfys)
        attackfys['HI Distance'] = df_fysical['HI Distance P90']
        attackfys['Distance'] = df_fysical['Distance P90']   
        attackfys_group = attackfys.groupby('Player').mean()
        attackfys_group.round(2)

        attack_grouped = attack.groupby('Player').mean()
        #attack_grouped['% Succesfull Actions'] = ((attack_grouped['Succesfull Actions'] / attack_grouped['Total Actions'])*100).round(2)
        attack_grouped['% Dribbles'] = ((attack_grouped['Succesfull Dribbles'] / attack_grouped['Dribbles'])*100).round(2)
        attack_grouped['% Passes'] = ((attack_grouped['Accurate Passes'] / attack_grouped['Passes'])*100).round(2)
        #attack_grouped['% Long Passes'] = (attack_grouped['Accurate Long Passes'] / attack_grouped['Long Passes']).round(2)
        attack_grouped['Rec./ Losses Ratio'] = (attack_grouped['Recoveries'] / attack_grouped['Losses']).round(2)
        attack_grouped['% Defensive Duels Won'] = ((attack_grouped['Defensive Duels Won'] / attack_grouped['Defensive Duels'])*100).round(2)
        attack_grouped['% Crosses'] = ((attack_grouped['Accurate Crosses'] / attack_grouped['Crosses'])*100).round(2)
        attack_grouped['% Aerial Duels Won'] = ((attack_grouped['Won Aerial Duels'] / attack_grouped['Aerial Duels'])*100).round(2)
        attack_grouped.drop(columns=['Accurate Crosses', 'Succesfull Actions', 'Total Actions', 'Dribbles', 'Succesfull Dribbles', 'Passes', 'Accurate Passes', 'Recoveries', 'Losses', 'Defensive Duels Won', 'Defensive Duels', 'Won Aerial Duels', 'Aerial Duels'],
                            inplace= True)
        Attackers = attack_grouped.merge(attackfys_group, how= 'left', on = 'Player')
        lijst_spelersgroep.append(Attackers)

    def centerback_radar(df_general, df_defense, df_attack, df_passing, df_fysical, bestandsnaam):
        naam_delen = os.path.splitext(bestandsnaam)[0].split()
        speler_naam = " ".join(naam_delen[6:8])
        attack = pd.DataFrame()
        mean_minutes = df_general['Minutes played'].mean()
        attack['Player'] = [bestandsnaam] * len(df_general)
        attack['Total Actions'] = df_general.iloc[:, 5]
        attack['Succesfull Actions'] = df_general.iloc[:, 6]
        #attack['% Succesfull Actions'] = np.nan
        #attack['Dribbles'] = df_attack.iloc[:, 13]
        #attack['Succesfull Dribbles'] = df_attack.iloc[:, 14]
        #attack['% Succesfull Dribbles'] = np.nan
        attack['Passes'] = df_passing.iloc[:, 5]
        attack['Accurate Passes'] = df_passing.iloc[:, 6]
        attack['% Accurate Passes'] = np.nan
        attack['Forward Passes'] = df_passing.iloc[:, 21]
        attack['Accurate Forward Passes'] = df_passing.iloc[:, 22]
        attack['% Accurate Forward Passes'] = np.nan
        attack['Long Passes'] = df_passing.iloc[:, 8]
        #attack['Accurate Long Passes'] = df_passing.iloc[:, 8]
        #attack['% Long Passes'] = np.nan
        attack['Interceptions'] = df_defense.iloc[:, 13]
        attack['Clearances'] = df_defense.iloc[:, 18]
        attack['Losses'] = df_defense.iloc[:, 14]
        attack['Recoveries'] = df_defense.iloc[:, 16]
        attack['Rec./ Losses Ratio'] = np.nan
        
        attack['Defensive Duels'] = df_defense.iloc[:, 5]
        attack['Defensive Duels Won'] = df_defense.iloc[:, 6]
        attack['% Defensive Duels Won'] = np.nan
        attack['Aerial Duels'] = df_defense.iloc[:, 7]
        attack['Won Aerial Duels'] = df_defense.iloc[:, 8]
        attack['% Won Aerial Duels'] = np.nan
        attackfys = pd.DataFrame()
        attackfys['Topspeed'] = df_fysical['PSV-99']
        attackfys['Accelerations'] = df_fysical['Count High Acceleration P90']
        attackfys['Player'] = [bestandsnaam] * len(attackfys)
        attackfys['Distance'] = df_fysical['Distance P90']
        attackfys_group = attackfys.groupby('Player').mean()
        attackfys_group.round(2)

        attack_grouped = attack.groupby('Player').mean()
        attack_grouped['% Accurate Forward Passes'] = ((attack_grouped['Accurate Forward Passes'] / attack_grouped['Forward Passes'])*100).round(2)
        #attack_grouped['% Succesfull Actions'] = ((attack_grouped['Succesfull Actions'] / attack_grouped['Total Actions'])*100).round(2)
        #attack_grouped['% Succesfull Dribbles'] = ((attack_grouped['Succesfull Dribbles'] / attack_grouped['Dribbles'])*100).round(2)
        attack_grouped['% Accurate Passes'] = ((attack_grouped['Accurate Passes'] / attack_grouped['Passes'])*100).round(2)
        #attack_grouped['% Long Passes'] = ((attack_grouped['Accurate Long Passes'] / attack_grouped['Long Passes'])*100).round(2)
        attack_grouped['Rec./ Losses Ratio'] = (attack_grouped['Recoveries'] / attack_grouped['Losses']).round(2)
        attack_grouped['% Defensive Duels Won'] = ((attack_grouped['Defensive Duels Won'] / attack_grouped['Defensive Duels'])*100).round(2)
        attack_grouped['% Won Aerial Duels'] = ((attack_grouped['Won Aerial Duels'] / attack_grouped['Aerial Duels'])*100).round(2)
        attack_grouped.drop(columns=[ 'Accurate Forward Passes','Forward Passes', 'Succesfull Actions', 'Total Actions', 'Passes', 'Accurate Passes', 'Losses', 'Defensive Duels Won', 'Defensive Duels', 'Won Aerial Duels', 'Aerial Duels'],
                            inplace= True)
        Attackers = attack_grouped.merge(attackfys_group, how= 'left', on = 'Player')
        lijst_spelersgroep.append(Attackers)
        
    def radar(df, lijst):
        lijst2 = list(reversed(lijst))
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            options1 = st.radio('Speler1', options=lijst)
        with col3:
            options2 = st.radio('Speler2', options=lijst2)

        test = df.loc[(df['Player'] == options1) | (df['Player'] == options2)]
        df1 = test.dropna(axis=1, how='any')

        numerieke_df = df.select_dtypes(include='number')
        gemiddelden = numerieke_df.mean()
        df_parameters = df.fillna(gemiddelden)
        
        df1.reset_index(drop=True, inplace= True)
        params = list(df1.columns)
        params = params[1:]
        params
        df2 = pd.DataFrame()
        df2 = df1.set_index('Player')
        
        ranges = []
        a_values = []
        b_values = []
        #st.markdown(df1[params])
        for x in params:
            if x == 'Topspeed':
                a = min(df_parameters[params][x])
                a = a * 0.96
            else:
                a = min(df_parameters[params][x])
                a = a * 0.90
            
            if x == 'Topspeed':
                b = max(df_parameters[params][x])
                b = b * 1.04
            else:
                b = max(df_parameters[params][x])
                b = b * 1.1
            
            ranges.append((a, b))
            a_values.append(a)
            b_values.append(b)

        #st.dataframe(a_values)
        player_1 = df1.iloc[0,0]
        player_2 = df1.iloc[1,0]

        for x in range(len(df1['Player'])):
            x = x - 1
            if df1.iloc[x, 0] == df1.iloc[0,0]:
                a_values = df1.iloc[x].values.tolist()
            if df1.iloc[x, 0] == df1.iloc[1,0]:
                b_values = df1.iloc[x].values.tolist()

        a_values = a_values[1:]
        b_values = b_values[1:]
        values = (a_values, b_values)

        title = dict(
        title_name=f'{player_1} (red)',
        title_color='#B6282F',
        title_name_2=f'{player_2} (blue)',
        title_color_2='#344D94',
        title_fontsize=15,
        subtitle_fontsize=11
        )

        radar = Radar()

        fig, ax = radar.plot_radar(ranges= ranges, params= params, values= values, 
                                radar_color=['red','blue'], 
                                title = title,
                                alphas = [0.3, 0.3],  
                                compare=True)
        fig.set_size_inches(12, 12)
        st.dataframe(df2)
        with col2:
            st.pyplot(fig)

        
    def inlezen_bestanden(lijst):
        for player in lijst:
            df_g = pd.read_excel(f"Player stats {player}.xlsx")
            df_d = pd.read_excel(f"Player stats {player} (1).xlsx")
            df_a = pd.read_excel(f"Player stats {player} (2).xlsx")
            df_p = pd.read_excel(f"Player stats {player} (3).xlsx")
            try:
                df_f = pd.read_csv(f"SkillCorner {player}.csv", encoding='latin1', sep=';')
            except:
                df_f = pd.read_csv(f"SkillCorner T. Tissoudali.csv", encoding='latin1', sep=';')
                df_f.drop(df_f.index, inplace=True)
            speler = player
            #st.dataframe(df_g)
            attacker(df_g, df_d, df_a, df_p, df_f, speler)
            if options == 'Spitsen': 
                attacker_radar(df_g, df_d, df_a, df_p, df_f, speler)
            if options == 'Centrale Middenvelders':
                Cmidfield_radar(df_g, df_d, df_a, df_p, df_f, speler)
            if options == 'Aanvallende Middenvelders':
                Amidfield_radar(df_g, df_d, df_a, df_p, df_f, speler)
            if options == 'Vleugel Aanvallers':
                Winger_radar(df_g, df_d, df_a, df_p, df_f, speler)
            if options == 'Vleugel Verdedigers':
                wingback_radar(df_g, df_d, df_a, df_p, df_f, speler)
            if options == 'Centrale Verdedigers':
                centerback_radar(df_g, df_d, df_a, df_p, df_f, speler)

    # Hoofdpagina met informatie/ en de navigatiebar
    col1,col2 = st.columns([8, 1.5])
    with col1:
        st.subheader('Technische Spelersanalyse')
    #with col2:
        #st.image("logo.png", width=150)
    st.markdown('Beschrijving van de app:')      
    st.markdown("- Met deze tool kan de data van verschillende spelers worden vergeleken over het seizoen 23/24 (Competitie, Beker en Europese matchen).")
    st.markdown("- Data van zowel WyScout voor technische informatie als SkillCorner voor fysieke data wordt gebruikt. Merk op dat de nauwkeurigheid van WyScout data niet altijd 100% is, neem dit in overweging bij genomen analyses. Beschrijving van alle gebruikte metrics is te vinden op https://dataglossary.wyscout.com")
    st.markdown("- Alle gegevens worden omgezet naar gemiddelden per 90 minuten")
    st.markdown("- Voor elke positie worden enkele spelers belicht, waaronder die van KAA Gent zelf, andere spelers van de JPL, alsook prospects voor KAA Gent.")
    st.markdown("- Deze objectieve weergave van spelersprestaties kan de subjectieve beoordelingen aanvullen, waardoor een meer alomvattend beeld ontstaat.")
    st.markdown("- Spelers kunnen altijd worden toegevoegd of verwijderd uit de app.")
    st.divider()
    st.write("")
    st.write("")

    options = st.sidebar.radio('Positie', options=['Spitsen', 'Vleugel Aanvallers', 'Aanvallende Middenvelders', 'Centrale Middenvelders', 'Vleugel Verdedigers', 'Centrale Verdedigers'])
    if options == 'Spitsen':
        radar_lijst = list_attackers
        inlezen_bestanden(radar_lijst)
        radar_lijst.append('Top 6 Striker JPL')
    if options == 'Centrale Middenvelders':
        radar_lijst = list_centralmidfield
        inlezen_bestanden(radar_lijst)
        radar_lijst.append('Top 6 Central Mid JPL')
    if options == 'Aanvallende Middenvelders':
        radar_lijst = list_attackingmidfield
        inlezen_bestanden(radar_lijst)
        radar_lijst.append('Top 6 AM JPL')
    if options == 'Vleugel Aanvallers':
        radar_lijst = list_wingers
        inlezen_bestanden(radar_lijst)
        radar_lijst.append('Top 6 Winger JPL')
    if options == 'Vleugel Verdedigers':
        radar_lijst = list_wingbacks
        inlezen_bestanden(radar_lijst)
        radar_lijst.append('Top 6 Back JPL')
        radar_lijst.append('Top 6 Wingback JPL')
    if options == 'Centrale Verdedigers':
        radar_lijst = list_centerbacks
        inlezen_bestanden(radar_lijst)
        radar_lijst.append('Top 6 Central Def JPL')

    general = pd.concat(lijst_spelersgroep)
    attack = pd.concat(lijst_attack)
    defense = pd.concat(lijst_defense)
    passing = pd.concat(lijst_passing)
    fysical = pd.concat(lijst_fysical)
    fysical = fysical.round(1)
    #st.dataframe(fysical)
    games = pd.concat(lijst_games)

    title_with_icon('⚽️', "Gespeelde matchen")
    st.markdown("Het aantal matchen waarvoor data werd verzameld per speler.")
    games = games.set_index('Player')
    spelers = games.reset_index()
    col1, col2 = st.columns([2.18, 4])
    with col1:
        speler = st.multiselect('spelers', options= spelers['Player'])
    if not speler:
        st.dataframe(games)
    else:
        games_sp = spelers[(spelers['Player'].isin(speler))]
        games_sp.set_index('Player', inplace=True)
        st.dataframe(games_sp)

    st.divider()

    title_with_icon('📋', "Technishe en Fysieke Data")
    'Overzicht van zowel de technische als de fysieke data van de geselecteerde spelers (Per 90 minuten).'
    col1, col2, col3, col4 = st.columns([1, 1.5, 1.5, 1.5])
    spelers = general.reset_index()
    options_4 = []
    with col1:
        options_2 = st.radio('Data', options=['Algemeen', 'Aanvallend', 'Passing', 'Verdedigend', 'Fysiek'])
    with col2:
        options_3 = st.multiselect('Spelers', options= spelers['Player'])
        if options_2 == 'Algemeen':
            options_4 = st.multiselect('Parameters', options= general.columns)
        if options_2 == 'Aanvallend':
            options_4 = st.multiselect('Parameters', options= attack.columns)
        if options_2 == 'Passing':
            options_4 = st.multiselect('Parameters', options= passing.columns)
        if options_2 == 'Verdedigend':
            options_4 = st.multiselect('Parameters', options= defense.columns)
        if options_2 == 'Fysiek':
            options_4 = st.multiselect('Parameters', options= fysical.columns)
    #with col3:
        

    options_4  
    #with col3:
    if options_2 == 'Algemeen':
        spelers = general.reset_index()
        if not options_3:
            if not options_4:
                st.dataframe(general)
            else:
                general_par = general[options_4]
                st.dataframe(general_par)
        else:
            general_pl = spelers[(spelers['Player'].isin(options_3))]
            general_pl.set_index('Player', inplace=True)
            if not options_4:
                st.dataframe(general_pl)
            else:
                general_par = general_pl[options_4]
                st.dataframe(general_par)    
    elif options_2 == 'Aanvallend':
        spelers = attack.reset_index()
        if not options_3:
            if not options_4:
                st.dataframe(attack, height= attack.shape[0]*37)
            else:
                attack_par = attack[options_4]
                st.dataframe(attack_par)
        else:
            attack_pl = spelers[(spelers['Player'].isin(options_3))]
            attack_pl.set_index('Player', inplace=True)
            if not options_4:
                st.dataframe(attack_pl)
            else:
                attack_par = attack_pl[options_4]
                st.dataframe(attack_par) 
    elif options_2 == 'Passing':
        spelers = passing.reset_index()
        if not options_3:
            if not options_4:
                st.dataframe(passing, height= attack.shape[0]*37)
            else:
                passing_par = passing[options_4]
                st.dataframe(passing_par)
        else:
            passing_pl = spelers[(spelers['Player'].isin(options_3))]
            passing_pl.set_index('Player', inplace=True)
            if not options_4:
                st.dataframe(passing_pl)
            else:
                passing_par = passing_pl[options_4]
                st.dataframe(passing_par)
    elif options_2 == 'Fysiek':
        spelers = fysical.reset_index()
        if not options_3:
            if not options_4:
                st.dataframe(fysical.round(1), height= fysical.shape[0]*37)
            else:
                fysical_par = fysical[options_4]
                st.dataframe(fysical_par)
        else:
            fysical_pl = spelers[(spelers['Player'].isin(options_3))]
            fysical_pl.set_index('Player', inplace=True)
            if not options_4:
                st.dataframe(fysical_pl)
            else:
                fysical_par = fysical_pl[options_4]
                st.dataframe(fysical_par)
    else:
        spelers = defense.reset_index()
        if not options_3:
            if not options_4:
                st.dataframe(defense, height= defense.shape[0]*37)
            else:
                defense_par = defense[options_4]
                st.dataframe(defense_par)
        else:
            defense_pl = spelers[(spelers['Player'].isin(options_3))]
            defense_pl.set_index('Player', inplace=True)
            if not options_4:
                st.dataframe(defense_pl)
            else:
                defense_par = defense_pl[options_4]
                st.dataframe(defense_par)

    st.divider()

    title_with_icon('📊', "Radar Charts")
    st.markdown("- Vergelijk de data van twee spelers via de radar chart.")
    st.markdown("- Vergelijk een speler met de gemiddelde data van een speler in de top 6 van de JPL + Kaa Gent.")

    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")

    attack = pd.concat(lijst_spelersgroep)
    attack.reset_index(inplace=True)
    if options == 'Spitsen':
        Top6 = attack.loc[attack['Player'].isin(['T. Tissoudali',  'G. Orban', 'H. Cuypers', 'G. Nilsson', 'K. Dolberg', 'T. Arokodare', 'I. Thiago', 'K. Denkey', 'D. Ayensa', 'V. Janssen', 'M. Amoura'])]
        Top6['Player'] = 'Top 6 Striker JPL'
        Top6 = Top6.groupby('Player').mean()
        Top6.reset_index(inplace=True)
        attack = pd.concat([attack, Top6])
    if options == 'Centrale Middenvelders':
        Top6 = attack.loc[attack['Player'].isin(['S. Kums', 'J. De_Sart', 'R. Onyedika', 'C. Nielsen', 'H. Vetlesen', 'P. Hrošovský', 'B. Heynen', 'M. Galarza', 'L. Lopes', 'H. Van_der_Bruggen', 'A. Vermeeren', 'M. Keita', 'A. Yusuf', 'T. Leoni', 'M. Rits', 'N. Sadiki', 'M. Rasmussen', 'C. Vanhoutte', 'J. Lazare'])]
        Top6['Player'] = 'Top 6 Central Mid JPL'
        Top6 = Top6.groupby('Player').mean()
        Top6.reset_index(inplace=True)
        attack = pd.concat([attack, Top6])
    if options == 'Centrale Verdedigers':
        Top6 = attack.loc[attack['Player'].isin(['T. Watanabe', 'I. Kandouss', 'J. Torunarigha', 'T. Alderweireld', 'J. Vertonghen', 'Z. Debast', 'B. Mechele', 'B. Popović', 'C. Burgess', 'C. Cuesta', 'J. Daland', 'J. Spileers', 'K. Machida', 'M. McKenzie', 'R. Sykes', 'Z. Van_den_Bosch'])]
        Top6['Player'] = 'Top 6 Central Def JPL'
        Top6 = Top6.groupby('Player').mean()
        Top6.reset_index(inplace=True)
        attack = pd.concat([attack, Top6])
    if options == 'Vleugel Aanvallers':
        Top6 = attack.loc[attack['Player'].isin(['T. Hazard', 'A. Nusa', 'M. Fofana', 'A. Dreyer', 'C. Ejuke', 'M. Balikwisha', 'A. Minda', 'T. Somers', 'J. Paintsil', 'A. Skov_Olsen', 'P. Zinckernagel'])]
        Top6['Player'] = 'Top 6 Winger JPL'
        Top6 = Top6.groupby('Player').mean()
        Top6.reset_index(inplace=True)
        attack = pd.concat([attack, Top6])
    if options == 'Vleugel Verdedigers':
        Top6 = attack.loc[attack['Player'].isin(['M. De_Cuyper', 'B. Meijer', 'Z. El_Ouahdi', 'J. Kayembe', 'G. Arteaga', 'D. Muñoz', 'H. Siquet', 'O. Wijndal', 'J. Bataille', 'K. Sardella', 'L. Augustinsson'])]
        Top6['Player'] = 'Top 6 Back JPL'
        Top6 = Top6.groupby('Player').mean()
        Top6.reset_index(inplace=True)
        attack = pd.concat([attack, Top6])
        Top6 = attack.loc[attack['Player'].isin(['A. Brown', 'M. Samoise', 'L. Lapoussin', 'A. Castro-Montes'])]
        Top6['Player'] = 'Top 6 Wingback JPL'
        Top6 = Top6.groupby('Player').mean()
        Top6.reset_index(inplace=True)
        attack = pd.concat([attack, Top6])
    if options == 'Aanvallende Middenvelders':
        Top6 = attack.loc[attack['Player'].isin(['H. Hong', 'B. El_Khannouss', 'H. Vanaken', 'J. Ekkelenkamp', 'C. Puertas'])]
        Top6['Player'] = 'Top 6 AM JPL'
        Top6 = Top6.groupby('Player').mean()
        Top6.reset_index(inplace=True)
        attack = pd.concat([attack, Top6])

    attack.reset_index(inplace=True, drop=True)
    attack = attack.round(2)
    #st.dataframe(attack)
    radar(attack, radar_lijst)
def FysiekeBenchmark():
            # Afronding vd dataframes
    def afronding(Dataframe):
        Dataframe.iloc[:, 2:8] = Dataframe.iloc[:, 2:8].round(0)
        Dataframe.iloc[:, 9:16] = Dataframe.iloc[:, 9:16].round(1)
        return Dataframe

    # Functie voor alle grafieken in de applicatie
    def Barplot(dataframe):
        selected_column = st.selectbox('Choose metric', dataframe.columns[3:16])

        plt.figure(figsize=(10,6))
        bars = plt.bar(dataframe['Position'], dataframe[selected_column])
        for bar in bars:
            yval = bar.get_height()
            #plt.text(bar.get_x() + bar.get_width() / 2, yval, ha= 'center', va='bottom')
            plt.text(bar.get_x() + bar.get_width() / 2, yval, str(yval), ha='center', va='bottom')
        if selected_column == 'Distance P90':
            y_min = 7000
            y_max = 14000
            plt.ylim(y_min, y_max)
        if selected_column == 'PSV-99':
            y_min = 20
            y_max = 40
            plt.ylim(y_min, y_max)
        plt.xlabel('Positie')
        plt.ylabel(selected_column)
        plt.title(f'{selected_column} per positie')
        st.pyplot(plt)

    def laatste_deel_na_spatie(naam):
        delen = naam.split()  # splits de naam op spaties
        return delen[-1]  # retourneert het laatste deel van de naam

    #functie voor wanneer files worden geupload ter vergelijking
    def compare(files):
        compare_prospects = pd.DataFrame()
        for file in files:
            prospect = pd.read_csv(file, encoding='latin1', sep=';')
            prospect.loc[prospect['Position'].isin(['LWB', 'RWB']), 'Position Group'] = 'Full Back'
            prospect.loc[prospect['Position'].isin(['CB', 'LCB', 'RCB']), 'Position Group'] = 'Central Defender'
            prospect.loc[prospect['Position'].isin(['AM', 'RM', 'LM', 'DM', 'RW', 'LW']), 'Position Group'] = 'Midfielder'
            prospect.loc[prospect['Position'].isin(['RW', 'LW']), 'Position Group'] = 'Winger'
            prospect.loc[prospect['Position'].isin(['CA', 'CF', 'LF', 'RF']), 'Position Group'] = 'Forward'
            dummy = prospect['Position Group']
            prospect.drop(columns=prospect.columns[1], inplace = True)
            prospect.drop(columns=prospect.columns[1], inplace = True)
            prospect.drop(columns=prospect.columns[-1], inplace = True)
            prospect.rename(columns={prospect.columns[0]: "Position", prospect.columns[1]: "Total Minutes Played"}, inplace=True)
            Prospect = prospect.groupby('Position').mean().reset_index()
            Prospect_sum = prospect.groupby('Position').sum().reset_index()
            Prospect['Total Minutes Played'] = Prospect_sum['Total Minutes Played']
            Prospect['Position Group'] = dummy
            Prospect.reset_index(drop=True, inplace=True)
            cols = list(Prospect.columns)
            Prospect = Prospect[[cols[0]] + [cols[-1]] + cols[1:15]]
            Prospect.reset_index(drop=True, inplace=True)
            Prospect = afronding(Prospect)
            #Prospect['Position'] = Prospect['Position'].apply(laatste_deel_na_spatie)              
            compare_prospects = pd.concat([compare_prospects, Prospect], axis=0)
        return compare_prospects

    #functie voor wanneer een bepaalde opstelling wordt aangeklikt    
    def formatie(dataframe):

        if options == '3-5-2':
            st.markdown('In total, 27 games have been recorded in the 3-5-2 formation.')
        elif options == '3-4-3':
            st.markdown('In total, 10 games have been recorded in the 3-4-3 formation.')
        else:
            st.markdown('In total, 4 games have been recorded in the 4-3-3 formation.')

        dataframe.drop(columns= dataframe.columns[0:3], inplace=True)
        dataframe.drop(columns= dataframe.columns[2:4], inplace=True)

        Per_Position_90 = dataframe.groupby('Position').mean()
        Sum_Per_Position_90 = dataframe.groupby('Position').sum()
        Per_Position_90['Total Minutes Played'] = Sum_Per_Position_90['Minutes Played']

        Per_Position_90 = Per_Position_90.reset_index()
        Per_Position_90.loc[Per_Position_90['Position'] == 'LW', 'Position'] = 'LWB'
        Per_Position_90.loc[Per_Position_90['Position'] == 'RW', 'Position'] = 'RWB'
        Per_Position_90.loc[Per_Position_90['Position'] == 'RM', 'Position'] = 'RCM'
        Per_Position_90.loc[Per_Position_90['Position'] == 'LM', 'Position'] = 'LCM'
        if options == '4-3-3':
            Per_Position_90.loc[Per_Position_90['Position'] == 'LWB', 'Position'] = 'LB'
            Per_Position_90.loc[Per_Position_90['Position'] == 'RWB', 'Position'] = 'RB'
        Per_Position_90.loc[Per_Position_90['Position'].isin(['LWB', 'RWB', 'LB', 'RB']), 'Position Group'] = 'Full Back'
        Per_Position_90.loc[Per_Position_90['Position'].isin(['CB', 'LCB', 'RCB']), 'Position Group'] = 'Central Defender'
        Per_Position_90.loc[Per_Position_90['Position'].isin(['AM', 'RM', 'LM', 'DM', 'RW', 'LW', 'RCM', 'LCM']), 'Position Group'] = 'Midfielder'
        Per_Position_90.loc[Per_Position_90['Position'].isin(['RW', 'LW']), 'Position Group'] = 'Winger'
        Per_Position_90.loc[Per_Position_90['Position'].isin(['CA', 'CF', 'LF', 'RF']), 'Position Group'] = 'Forward'

        cols = list(Per_Position_90.columns)
        Per_Position_90 = Per_Position_90[[cols[0]] + [cols[-1]] + [cols[-2]] + cols[2:15]]

        position_ranking_map =  {
        'RWB': 1, 'RB': 2, 'RCB': 3, 'CB': 4, 'LCB': 5, 'LWB': 6, 'LB': 7,
        'RW': 8, 'RCM': 9, 'LCM': 10, 'LW': 11, 'AM': 12,
        'RF': 13, 'CF': 14, 'LF': 15
        }
        Per_Position_90['Ranking'] = Per_Position_90['Position'].map(position_ranking_map)
        Per_Position_90.sort_values(by='Ranking', inplace = True)
        Per_Position_90.drop(columns=['Ranking'], inplace = True)

        Per_Position_90 = afronding(Per_Position_90)

        selected_role = st.sidebar.multiselect('Position Group', ['Forward', 'Winger', 'Midfielder', 'Full Back', 'Central Defender'])
        df_selected_role = Per_Position_90[(Per_Position_90['Position Group'].isin(selected_role))]

        if not selected_role:
            st.dataframe(Per_Position_90, hide_index = True)
            st.markdown("<h4>Visualization</h4>", unsafe_allow_html=True)
            Barplot(Per_Position_90)

        else:
            selected_position = st.sidebar.multiselect('Position', df_selected_role['Position'].unique())
            df_selected_position = df_selected_role[(df_selected_role['Position'].isin(selected_position))]
            if not selected_position:
                df_selected_role.reset_index(drop=True, inplace=True)
                df_selected_role = df_selected_role.rename_axis('Rank', axis=0)
                df_selected_role.index = df_selected_role.index + 1
                st.dataframe(df_selected_role, hide_index= True)
                st.markdown("<h4>Visualization</h4>", unsafe_allow_html=True)
                Barplot(df_selected_role)

            else:
                df_selected_position.reset_index(drop=True, inplace=True)
                df_selected_position = df_selected_position.rename_axis('Rank', axis=0)
                df_selected_position.index = df_selected_position.index + 1
                st.dataframe(df_selected_position, hide_index= True)
                st.markdown("<h4>Compare with prospect(s)</h4>", unsafe_allow_html=True)
                st.markdown('In this section, you can compare the stats of different prospects with the benchmarks of KAA Gent. To do this, export the player data from SkillCorner and upload the file below.')
                st.write("**Export player data from SkillCorner with the following settings:**")
                st.write("       - Ensure that **'P90'** is selected.")
                st.write("       - In **General** settings, only **'Player'**, **'Match'** and **'Position'** should be selected.")
                st.write("       - In **Metrics**, everything should be selected except **'M/min'**.")
                uploaded_files = st.file_uploader('upload player file(s) here', accept_multiple_files=True)

                if uploaded_files:
                    compare_prospects = compare(uploaded_files)
                    df_compare = pd.concat([df_selected_position, compare_prospects])
                    st.dataframe(df_compare, hide_index=True)
                    df_compare['Position'] = df_compare['Position'].apply(laatste_deel_na_spatie) 
                    Barplot(df_compare)

    # Hoofdpagina met informatie/ en de navigatiebar
    st.subheader('KAA Gent Physical Performance Benchmarking Tool')
    st.markdown('Within this tool, a benchmark of the average physical data of KAA Gent can be found. The data spans across the 23/24 season, excluding the playoffs and friendly matches. Here\'s what you can do:')      
    st.markdown("- View the average physical output for KAA Gent players, categorized by formation.")
    st.markdown("- Further filter by position within each formation.")
    st.markdown("- Compare the physical output using visualizations.")
    st.markdown("- Compare the physical output of KAA Gent players with prospects from other teams.")
    st.divider()
    st.write("")
    st.write("")
    st.markdown("<h4>Physical statistics by position</h4>", unsafe_allow_html=True)
    options = st.sidebar.radio('Formations', options=['3-5-2', '3-4-3', '4-3-3'])

    if options == '4-3-3':
        df = pd.read_csv("433__P90.csv", encoding='latin1', sep=';')
        formatie(df)
    if options == '3-5-2':
        df = pd.read_csv("352___P90.csv", encoding='latin1', sep=';')
        formatie(df)
    if options == '3-4-3':
        df = pd.read_csv("343P90.csv", encoding='latin1', sep=';')
        formatie(df)

options = st.sidebar.radio('Pages', options=['Technische Spelersanalyse', 'Fysieke Scoutingstool', 'Fysieke Benchmarking'])

if options == 'Technische Spelersanalyse':
    radarcharts()
elif options == 'Fysieke Benchmarking':
    FysiekeBenchmark()
else:
    FysiekeDashboard()
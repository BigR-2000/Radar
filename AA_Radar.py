import pandas as pd
import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
#from mplsoccer import Radar
from soccerplots.radar_chart import Radar
#from Benchmarking_functions import hallo

list_attackers = ['T. Tissoudali', 'L. Depoitre', 'G. Orban', 'H. Cuypers', 'A. Lind', 'A. Gudjohnsen', 'M. Breunig', 'M. Biereth', 'G. Nilsson', 'K. Dolberg', 'T. Arokodare', 'I. Thiago', 'K. Denkey', 'A. Ementa', 'D. Ayensa', 'V. Janssen', 'M. Amoura', 'T. Barry', 'I. Matanović', 'F. Mayulu', 'G. Borrelli', 'R. Durosinmi'] #'M. Biereth', 'A. Gudjohnsen''M. Fernandez-Pardo''Cho_Gue-Sung', 'N. Milanovic', 'D. Maldini','A. Crnac''P. Ratkov', 'F. Girotti', 'Y. Salech', 'R. Čaks','B. Barišić', 'B. Nsimba', ]
list_centralmidfield = ['S. Kums', 'J. De_Sart', 'O. Gandelman', 'P. Gerkens', 'R. Onyedika', 'C. Nielsen', 'H. Vetlesen', 'P. Hrošovský', 'B. Heynen', 'M. Galarza', 'L. Lopes', 'H. Van_der_Bruggen', 'A. Vermeeren', 'M. Keita', 'A. Yusuf', 'T. Leoni', 'M. Rits', 'N. Sadiki', 'M. Rasmussen', 'C. Vanhoutte', 'L. Amani', 'A. Kadri',  'O. Højlund', 'P. Berg', 'P. Aaronson', 'O. Sørensen', 'T. Rigo', 'A. Bernede'] #'C. Rodri', , 'S. Resink','S. Esposito', 'J. Bakke', 'R. Pukštas', 'A. Morris',  'G. Busio',
list_attackingmidfield = ['H. Hong', 'A. Hjulsager', 'B. El_Khannouss', 'H. Vanaken', 'J. Ekkelenkamp', 'C. Puertas', 'A. Omgba']
list_wingers = ['D. Yokota', 'M. Fofana', 'M. Sonko', 'A. Sanches', 'J. Steuckers', 'T. Hazard', 'A. Nusa',  'A. Dreyer', 'C. Ejuke', 'M. Balikwisha', 'A. Minda', 'T. Somers', 'J. Paintsil', 'A. Skov_Olsen', 'P. Zinckernagel']
list_wingbacks = [ 'M. Samoise', 'A. Brown', 'N. Fadiga', 'L. Lapoussin', 'A. Castro-Montes', 'M. De_Cuyper', 'B. Meijer', 'Z. El_Ouahdi', 'J. Kayembe', 'G. Arteaga', 'D. Muñoz', 'H. Siquet', 'O. Wijndal', 'J. Bataille', 'K. Sardella', 'L. Augustinsson']#, 'I. Camara',
list_centerbacks = ['S. Mitrović', 'J. Torunarigha', 'I. Kandouss',  'T. Watanabe', 'A. Bright', 'N. Abbey', 'Z. Debast', 'T. Alderweireld', 'J. Vertonghen', 'B. Mechele', 'B. Popović', 'C. Burgess', 'C. Cuesta', 'J. Daland', 'J. Spileers', 'K. Machida', 'M. McKenzie', 'R. Sykes', 'Z. Van_den_Bosch', 'A. N\'Diaye', 'T. Cissokho', 'H. Petrov', 'M. Nadé', 'S. Karič']#'José_Marsà','J. Cordoba', 'L. Lochoshvili', 'D. Cornelius','U. Bilbao', , 'P. Awuku', 'E. Cobbaut', 'A. Filin', 'J. Rasmussen', 'S. Kotto', 'S. Kotto2', 'P. Bochniewicz',, 'A. Filin''M. Mbow',, 'E. Cobbaut'
st.set_page_config(page_title = 'Technical Benchmark Kaa Gent',
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
    attack['Player'] = ([speler_naam] * len(df_attack))
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
    passing['Player'] = ([speler_naam] * len(df_passing))
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
    defense['Player'] = ([speler_naam] * len(df_defense))
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
    attack['% Succesfull Actions'] = np.nan
    attack['Goals'] = df_attack.iloc[:, 5]
    attack['xG'] = df_attack.iloc[:, 9]
    attack['Goals - xG'] = np.nan
    attack['Goals per Shot Ratio'] = np.nan
    attack['Assists'] = df_attack.iloc[:, 6]
    attack['Passes'] = df_passing.iloc[:, 5]
    attack['Accurate Passes'] = df_passing.iloc[:, 6]
    #attack['Progressive Runs'] = df_attack.iloc[:, 19]
    attack['% Accurate Passes'] = np.nan
    attack['Shots'] = df_attack.iloc[:, 7]
    #attack['Shots On Target'] = df_attack.iloc[:, 8]
    #attack['% Shots On Target'] = np.nan 
    attack['Dribbles'] = df_attack.iloc[:, 13]
    attack['Succesfull Dribbles'] = df_attack.iloc[:, 14]
    attack['% Dribbles'] = np.nan

    attack['Offensive Duels'] = df_attack.iloc[:, 15]
    attack['Won Offensive Duels'] = df_attack.iloc[:, 16]
    attack['% Won Offensive Duels'] = np.nan
    attack['Aerial Duels'] = df_defense.iloc[:, 7]
    attack['Won Aerial Duels'] = df_defense.iloc[:, 8]
    attack['% Won Aerial Duels'] = np.nan
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
    attack_grouped['% Succesfull Actions'] = ((attack_grouped['Succesfull Actions'] / attack_grouped['Total Actions']) * 100).round(2)
    #attack_grouped['% Shots On Target'] = ((attack_grouped['Shots On Target'] / attack_grouped['Shots']) * 100).round(2)
    attack_grouped['% Dribbles'] = ((attack_grouped['Succesfull Dribbles'] / attack_grouped['Dribbles']) * 100).round(2)
    attack_grouped['% Won Offensive Duels'] = ((attack_grouped['Won Offensive Duels'] / attack_grouped['Offensive Duels']) * 100).round(2)
    attack_grouped['% Accurate Passes'] = ((attack_grouped['Accurate Passes'] / attack_grouped['Passes']) * 100).round(2)
    attack_grouped['% Won Aerial Duels'] = ((attack_grouped['Won Aerial Duels'] / attack_grouped['Aerial Duels']) * 100).round(2)
    #attack_grouped['Rec./ Losses Ratio'] = (attack_grouped['Recoveries'] / attack_grouped['Losses']).round(2)
    attack_grouped['Goals per Shot Ratio'] = attack_grouped['Goals'] / attack_grouped['Shots']
    attack_grouped['Goals - xG'] = attack_grouped['Goals'] - attack_grouped['xG']
    attack_grouped.drop(columns=['Total Actions', 'Succesfull Actions', 'Dribbles', 'Passes', 
                                'Accurate Passes', 'Offensive Duels', 'Won Offensive Duels', 'Aerial Duels', 'Won Aerial Duels', 'Losses', 'Recoveries', 'Shots'],
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
    attack['% Succesfull Actions'] = np.nan
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
    attack_grouped['% Succesfull Actions'] = ((attack_grouped['Succesfull Actions'] / attack_grouped['Total Actions'])*100).round(2)
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
    attack['% Succesfull Actions'] = np.nan
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
    attack_grouped['% Succesfull Actions'] = ((attack_grouped['Succesfull Actions'] / attack_grouped['Total Actions'])*100).round(2)
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
    attack['% Succesfull Actions'] = np.nan
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
    attack_grouped['% Succesfull Actions'] = ((attack_grouped['Succesfull Actions'] / attack_grouped['Total Actions'])*100).round(2)
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
    attack['% Succesfull Actions'] = np.nan
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
    attack_grouped['% Succesfull Actions'] = ((attack_grouped['Succesfull Actions'] / attack_grouped['Total Actions'])*100).round(2)
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
    attack['% Succesfull Actions'] = np.nan
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
    attack_grouped['% Succesfull Actions'] = ((attack_grouped['Succesfull Actions'] / attack_grouped['Total Actions'])*100).round(2)
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
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        options1 = st.radio('Speler1', options=lijst)
    with col3:
        options2 = st.radio('Speler2', options=lijst)

    bestandsnaam1 = f"Skillcorner {options1}.csv"
    bestandsnaam2 = f"Skillcorner {options2}.csv"
    if options1.startswith('Top'):
        bestandsnaam1 = "Skillcorner G. Orban.csv"
    if options2.startswith('Top'):
        bestandsnaam2 = "Skillcorner G. Orban.csv"
    

    try:
        dfsp1 = pd.read_csv(bestandsnaam1, encoding='latin1', sep=';')
    except:
        st.markdown(f"Geen fysieke data beschikbaar voor {options1}.")
        te_verwijderen_kolommen = ['Distance', 'Total Distance', 'Sprints', 'Topspeed', 'PSV-99', 'HI Distance', 'High Accelerations', 'Accelerations']
        df = df.drop(columns=[kolom for kolom in te_verwijderen_kolommen if kolom in df])
        
    try:
        dfsp2 = pd.read_csv(bestandsnaam2, encoding='latin1', sep=';')
    except:
        st.markdown(f"Geen fysieke data beschikbaar voor {options2}.")
        te_verwijderen_kolommen = ['Distance', 'Total Distance', 'Sprints', 'Topspeed', 'PSV-99', 'HI Distance', 'High Accelerations', 'Accelerations']
        df = df.drop(columns=[kolom for kolom in te_verwijderen_kolommen if kolom in df])
        

    #opzetten dummy df voor het regelen van de ranges voor de radar chart zonder issues
    numerieke_df = df.select_dtypes(include='number')
    gemiddelden = numerieke_df.mean()
    df_parameters = df.fillna(gemiddelden)

    df1 = df.loc[(df['Player'] == options1) | (df['Player'] == options2)]
    df1.reset_index(drop=True, inplace= True)
    params = list(df1.columns)
    params = params[1:]
    params
    df2 = pd.DataFrame()
    df2 = df1.set_index('Player')
    
    #tabeleke = df1
    #tabeleke.set_index('Player', inplace=True)
    st.dataframe(df2)
    ranges = []
    a_values = []
    b_values = []
    #st.markdown(df1[params])
    for x in params:
        if x == 'Topspeed':
            a = min(attack[params][x])
            a = a * 0.96
        else:
            a = min(attack[params][x])
            a = a * 0.90
        
        if x == 'Topspeed':
            b = max(attack[params][x])
            b = b * 1.04
        else:
            b = max(attack[params][x])
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
    
    #radar = Radar(background_color="#121212", patch_color="#28252C", label_color="#FFFFFF",
              #range_color="#FFFFFF")
    #st.dataframe(ranges)
    fig, ax = radar.plot_radar(ranges= ranges, params= params, values= values, 
                            radar_color=['red','blue'], 
                            title = title,
                            alphas = [0.3, 0.3],  
                            compare=True)
    fig.set_size_inches(12, 12)
   
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
    st.subheader('Prestatie Analyse Tool KAA Gent')
#with col2:
    #st.image("logo.png", width=150)
st.markdown('Beschrijving van de app:')      
st.markdown("- Met deze tool kan de data van verschillende spelers worden vergeleken over het seizoen 23/24 (Competitie, Beker en Europese matchen).")
#st.markdown("- Further filter by position within each formation.")
st.markdown("- Deze objectieve weergave van spelersprestaties kan de subjectieve beoordelingen aanvullen, waardoor een meer alomvattend beeld ontstaat.")
st.markdown("- Data van zowel WyScout voor technische informatie als SkillCorner voor fysieke data wordt gebruikt. Merk op dat de nauwkeurigheid van WyScout data niet altijd 100% is, neem dit in overweging bij genomen analyses.")
st.markdown("- Alle gegevens worden omgezet naar gemiddelden per 90 minuten")
st.markdown("- Voor elke positie worden enkele spelers belicht, waaronder die van KAA Gent zelf, andere spelers van de JPL, alsook prospects voor KAA Gent.")
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

general = pd.concat(lijst_general)
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


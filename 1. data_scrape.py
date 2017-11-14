# TO SCRAPE THE DATA FROM THE WEB #

import json
import requests
from bs4 import BeautifulSoup
import pprint as pp
import time
from openpyxl import load_workbook
import pprint as pp

#returns a list of list of tuples.
#each tuple is in the form (idStart, idEnd). First tuple is 
#for 2015-2016 season, last tuple is for 2000 - 2001 season 
#each list of tuple represents the games from the following leagues
# in the order: Spain, England, Germany, Italy, France
def readGameWebIDs(excelFile= 'game_ids.xlsx'):
	mywb = load_workbook(excelFile)
	myws = mywb['Sheet1']

	colStart = ord('c')
	colEnd = ord('l') + 1
	rowStart = 3
	rowEnd = 19 #exclusive. i.e. actually ends at 18

	mydata = list()
	for col in range(colStart,colEnd,2):
		colLeft = col
		colRight = col + 1
		mylist = list()
		for row in range(rowStart,rowEnd,1):
			cellLeftAddr = chr(colLeft) + str(row)
			cellRightAddr = chr(colRight) + str(row)
			valueLeft = myws[cellLeftAddr].value
			valueRight = myws[cellRightAddr].value
			valTuple = (valueLeft,valueRight)
			mylist.append(valTuple)
		mydata.append(mylist)

	return mydata


#return bs object
def pullSoccerPage(linkName, outfileName):
	#response.content is in bytes
	response = requests.get(linkName)
	if response.status_code != 200:
		raise Exception("requests.get failed")

	outfile = open("%s.html" % (outfileName),'w')
	data_raw = BeautifulSoup(response.content,'lxml')
	outfile.write(data_raw.prettify())
	outfile.close()


#returns a list of dictionaries for all matches in results table 
#for the webpage's default group.
def getResultsTable(sourceFileName):
	source = open("%s.html" % (sourceFileName),"r")
	data_raw = BeautifulSoup(source,'lxml')
	url_top = "http://www.bdfutbol.com/en"
	# taulabdf is for all tables.. need to find results table
	# div for "Results" table
	results_div = data_raw.find("div",{"id":"jornada_classi"}) 
	results_table = results_div.table.find_all("tr")
	all_results = list()
	for row in results_table:
		match_info = dict()
		row_list = row.find_all("td")
		match_date = row_list[0].get_text()
		team_1 = row_list[1].get_text()
		team_1_score = row_list[2].get_text()
		team_2_score = row_list[3].get_text()
		team_2 = row_list[4].get_text()
		match_href = row.find('a')['href'][2:] #2:-1 strips ..
		match_info['date'] = match_date.strip()
		match_info['team1'] = team_1.strip()
		match_info['team2'] = team_2.strip()
		match_info['team1score'] = int(team_1_score.strip())
		match_info['team2score'] = int(team_2_score.strip())
		match_info['url'] = url_top + match_href
		all_results.append(match_info)
		# pp.pprint(match_info)
		# print('-----')
		# print(i.find_all("td")[0])
	
	source.close()
	return all_results

#sub function to get match information
def getMatchTeamEvents(data_raw,teamType):
	if teamType == 'home':
		searchParam = 'eqcasa'
	elif teamType == 'away':
		searchParam = 'eqfora'
	#get home team info first... Assume home team table is 2nd element
	team_div = data_raw.find_all("div",{"class":searchParam})[1]
	teamEvents = team_div.find_all("div",{"class":"cosa"})
	
	num_subout = 0
	num_yc = 0
	num_goal = 0
	num_subin = 0
	num_pengoal = 0
	num_owngoal = 0
	num_rc = 0

	subin_times = list()
	subout_times = list()
	yc_times = list()
	rc_times = list()
	goal_times = list()
	pengoal_times = list()
	owngoal_times = list()

	for event_i in teamEvents:
		event_i_child = event_i.find("div")
		event_i_type = event_i_child['class'][0].strip().lower()
		event_i_time = event_i.get_text().strip()
		try:
			event_i_time = int(event_i_time)
		except ValueError:
			event_i_time = None
		#sub out event
		if event_i_type == 'surt':
			subout_times.append(event_i_time)
			num_subout +=1
		#yellow card event	
		elif event_i_type == 'tg':
			yc_times.append(event_i_time)
			num_yc +=1 
		#goal event
		elif event_i_type == 'g':
			goal_times.append(event_i_time)
			num_goal += 1
		#sub in event
		elif event_i_type == 'entra':
			subin_times.append(event_i_time)
			num_subin += 1
		#penalty goal event
		elif event_i_type == 'gp':
			pengoal_times.append(event_i_time)
			num_pengoal += 1
		#own goals
		elif event_i_type == 'gpp':
			owngoal_times.append(event_i_time)
			num_owngoal += 1
		#red cards
		elif event_i_type == 'tv':
			rc_times.append(event_i_time)
			num_rc += 1
		else:
			raise Exception("event not recognized : %s \n" % (event_i_type))

	teamEvents_dict = dict()
	teamEvents_dict['yellowCards'] = yc_times
	teamEvents_dict['redCards'] = rc_times
	teamEvents_dict['subIns'] = subin_times
	teamEvents_dict['subOuts'] = subout_times
	teamEvents_dict['goals'] = goal_times
	teamEvents_dict['penaltyGoals'] = pengoal_times
	teamEvents_dict['ownGoals'] = owngoal_times

	return teamEvents_dict

#from html file, scrape all required data and return
#a dictionary of match information
def getMatchInfo(sourceFileName):
	source = open("%s.html" % (sourceFileName),"r")
	data_raw = BeautifulSoup(source,"lxml")

	#main dictionary to return
	match_info = dict()
	
	teamName_div = data_raw.find_all("div",{"class":"nom"})
	count = 1
	#get team names
	#assumes there are only 2 teams in teamName_div
	for team in teamName_div:
		teamName = team.get_text().strip().lower()
		if count == 1:
			match_info['homeTeam'] = teamName
		else:
			match_info['awayTeam'] = teamName
		count+=1
	count = 0
	#get match score and date info
	match_summary = data_raw.find("title")
	match_summary_str = match_summary.get_text().strip()
	homeScore_list = match_summary_str.split('(')
	#homeScore is assumed to be the 0th element of the string which is 
	#the 1st element
	homeScore = int(homeScore_list[1][0])
	
	awayScore_list = match_summary_str.split(')')
	#awayScore is assumed to be the LAST element of the first string
	awayScore = int(awayScore_list[0][-1])
	
	matchDate_list = match_summary_str.split(' ')
	matchDate = matchDate_list[-1]

	match_info['homeScore'] = homeScore
	match_info['awayScore'] = awayScore
	match_info['matchDate'] = matchDate
	
	#get stadium name
	match_stadium_div = data_raw.find('div',{'class':'info'})
	match_round_list = match_stadium_div.get_text().strip().split(' ')
	match_round = int(match_round_list[1])

	match_stadium_str = match_stadium_div.get_text().strip()
	stadium_start_idx = match_stadium_str.find('(') + 1
	#-1 is for last element in list, :-1 for string up till char before ")"
	match_stadium = match_stadium_str[stadium_start_idx:-1] 

	match_info['round'] = match_round
	match_info['stadium'] = match_stadium.lower()
	#get each team events
	homeTeamEvents = getMatchTeamEvents(data_raw,'home')
	awayTeamEvents = getMatchTeamEvents(data_raw,'away')

	match_info['homeEvents'] = homeTeamEvents
	match_info['awayEvents'] = awayTeamEvents
	# pp.pprint(match_info)
	return match_info

#pulls data from website and converts it into a dictionary
#all dictionaries are stored in a large json file
def getLeagueData(idStart,idEnd,leagueCountry,leagueYear):
	id_start = idStart #G = 602250 #609700
	id_end = idEnd + 1#602555 + 1 #id_start + 10
	link_top ="http://www.bdfutbol.com/en/p/p.php?id=" 
	leagueIDbase = leagueCountry[0].upper()
	filename_top = "data/%s" % (leagueIDbase)
	leagueName = leagueCountry + str(leagueYear)
	outfilename = 'league_data/%s.json' % (leagueName)

	print('idStart: %d, idEnd: %d, filename_top: %s, outfilename: %s \n'\
		%(idStart,idEnd,filename_top,outfilename))
	count = 1
	allLeagueGames  = dict()
	outfile = open(outfilename,'w')
	time_start = time.time()
	for idx in range(id_start,id_end,1):
		link_i = link_top + str(idx)
		filename_i = filename_top + str(idx)
		leagueGameID = leagueIDbase + str(idx)

		pullSoccerPage(link_i,filename_i)
		currLeagueGame = getMatchInfo(filename_i)
		
		allLeagueGames[leagueGameID] = currLeagueGame
		
		time_curr = time.time()
		time_run = time_curr - time_start

		print('\n ---------run time = %f ----id = %d------ \n '  % (time_run,idx))
		count += 1


	json.dump(allLeagueGames,outfile,indent=4,sort_keys=True)
	outfile.close()
	time_end = time.time()
	time_elapsed = time_end - time_start
	print('getLeagueData complete for %s. Time elapsed = %f' \
		% (leagueName, time_elapsed))
	return 


#search through excel file of game_id links, then pull data from internet 
def main():
	leagueOrder = ['spain','england','germany','italy','france']
	leagueYears = [i for i in range(2015,2007,-1)]
	allIDs = readGameWebIDs('game_ids.xlsx')
	numLeagues = len(leagueOrder)
	numYears = len(leagueYears)
	#iterate through league countries (0 to 4)
	for league_idx in range(0,numLeagues-1,1):
		currLeagueCountry = leagueOrder[league_idx]
		#iterate over leagueyears 0 to numleagues. change to 3 to numleagues
		for year_idx in range(3,4,1):
			currYear = leagueYears[year_idx]
			leagueI_idStart = allIDs[league_idx][year_idx][0]
			leagueI_idEnd = allIDs[league_idx][year_idx][1]
			print("%d \t %s. idStart = %d, idEnd = %d" \
				%(currYear,currLeagueCountry,leagueI_idStart,leagueI_idEnd))
			getLeagueData(leagueI_idStart,leagueI_idEnd,currLeagueCountry,currYear)

# main()
	
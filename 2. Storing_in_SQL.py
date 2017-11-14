#TO CREATE A DATABASE OF ALL THE GAMES IN SQL #

# creating a list of years for which we have data
years=['2010', '2011','2012','2013','2014','2015']
countries=['germany','england','italy','spain','france']
import json
import pymysql

for country in countries:
	for year in years:
		file_dir="C:\\Columbia\\StudyMaterial\\1. Data Analytics\\Project\\2. Data\\league_data\\"+country+year+".json"
	# Reading the json file that is stored after scraping
		test=open(file_dir).read()
		test=test.replace("'","")
		test=test.replace("\\","")
		test=json.loads(test)
	# Connecting to MySql
		db= pymysql.connect("localhost","root","!Prema407","data_analytics_project",autocommit=True)

		cursor=db.cursor()
		cursor.execute('drop table soccer_db_'+country+year+'')
		cursor.execute('create table soccer_db_'+country+year+'(country varchar(20), match_id varchar(20),match_Date varchar(20), stadium varchar(500),home_team varchar(50), away_team varchar(50), event varchar(20), time varchar(10))')
	# Inserting each event that we have into a record
	# For example - If there were 2 yellow cards - one at 60th and the other at 69th minute,
	# There will be two records in the SQL table
		for thing in test:    
			for event in test[thing]["homeEvents"]:
				for time in test[thing]["homeEvents"][event]:
				# it runs throught he dictionary and inserts the corresponding event and the time at which it happened into the respective columns in SQL
					sql = 'INSERT INTO soccer_db_'+country+year+' (country, match_Date, stadium, match_id, home_team, away_team, event, time) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)' % ("'"+country+"'","'"+test[thing]["matchDate"]+"'","'"+test[thing]["stadium"]+"'","'"+thing+"'","'"+test[thing]["homeTeam"]+"'", "'"+test[thing]["awayTeam"]+"'", "'home_"+event+"'", "'"+str(time)+"'")
					cursor.execute(sql)
			for event in test[thing]["awayEvents"]:
				for time in test[thing]["awayEvents"][event]:
				# The loop runs twice - one for home team events and one for away team. this is because the data is available in different places in the web
					sql = 'INSERT INTO soccer_db_germany'+year+' (country, match_Date, stadium, match_id, home_team, away_team, event, time) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)' % ("'"+country+"'","'"+test[thing]["matchDate"]+"'","'"+test[thing]["stadium"]+"'","'"+thing+"'","'"+test[thing]["homeTeam"]+"'", "'"+test[thing]["awayTeam"]+"'", "'away_"+event+"'", "'"+str(time)+"'")
					cursor.execute(sql)
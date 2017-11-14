from flask import Flask, session, render_template, url_for, request
from flask_wtf import FlaskForm
from wtforms import TextField, IntegerField, SelectField
from wtforms.fields.html5 import DateField
import numpy as np
import pandas as pd
import pandas.rpy.common as com
import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
import time
import matplotlib.pyplot as plt
import os
import pydotplus
import matplotlib 
import sklearn
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score

#Choosing CRAN mirror for integrating R into our program
utils = rpackages.importr("utils")
utils.chooseCRANmirror(ind=1)
classPacknames = ("class")
utils.install_packages(StrVector(classPacknames))

app = Flask(__name__)
app.secret_key = "abcdefg12345"

#Class 'dropFrom' defines all the parameters entered by the user
class dropForm(FlaskForm):
	scoreTime = SelectField("Time in game",coerce = int,choices =[(5,5), (10,10), \
		 (15,15),(20,20),(25,25),(30,30),(35,35),(40,40),(45,45),(50,50),(55,55),(60,60),(65,65),(70,70),(75,75),(80,80),(85,85)])
	homeTeam = TextField("Name of home team")
	awayTeam = TextField("Name of away team")
	homeScore = IntegerField("Goals scored by home team")
	awayScore = IntegerField("Goals scored by away team")
	homeYC = IntegerField("Yellow cards earned by home team")
	awayYC = IntegerField("Yellow cards earned by away team")
	homeRC = IntegerField("Red cards earned by home team")
	awayRC = IntegerField("Red cards earned by away team")
	homeSubs = IntegerField("Number of substitutions by home team")
	awaySubs = IntegerField("Number of substitutions by away team")
	model_type = SelectField("Choose a model", coerce = str, choices = [("Logistic_Regression", "Logistic Regression"), \
		("KNN", "K-Nearest Neighbors"), ("Decision_Trees", "Decision Tree Model")])

#returns a pandas dataframe of metrics for the team "teamName"
def getMetrics(teamName,home = True):
	if home:
		fileName = "home_metrics.csv"
		metricsDF = pd.read_csv(fileName)
		targetRow = metricsDF.ix[metricsDF.home_team.values == teamName]
		
		if not targetRow.empty:
			h_score_gd = float(targetRow['home_score_gd'])
			h_pct_win = float(targetRow['home_percent_win'])
			h_pct_pts = float(targetRow['home_percent_points'])
			return [h_score_gd,h_pct_win,h_pct_pts]
		else:
			return [0,0,0]
	else:
		fileName = "away_metrics.csv"
		metricsDF = pd.read_csv(fileName)
		targetRow = metricsDF.ix[metricsDF.away_team.values == teamName]
		
		if not targetRow.empty:
			a_score_gd = float(targetRow['away_score_gd'])
			a_pct_win = float(targetRow['away_percent_win'])
			a_pct_pts = float(targetRow['away_percent_points'])
			return [a_score_gd,a_pct_win,a_pct_pts]
		else:
			return [0,0,0]

#returns values from an r object dataframe
def extractRvalues(rObjVec):
	allRows = list()
	for item in rObjVec:
		rowI = list()
		for j in item:
			rowI.append(j)
		allRows.append(rowI)
	return allRows

#returns current date and time as a string, for use 
#in naming plots
def getDateTimeStr():
	dateToday = time.strftime("%d_%m_%Y")
	timeNow = time.strftime("%H_%M_%S")
	retStr = dateToday + "_" + timeNow
	return(retStr)

#Function runs multinomial logistic regression
def run_log_reg(homeTeam,awayTeam,hYC_i,h_Subs_i,h_RC_i,h_score_gd_i,h_pct_win_i,\
	h_pct_pts_i,aYC_i,a_Subs_i,a_RC_i,a_score_gd_i,a_pct_pts_i,\
	a_pct_win_i,h_score_i,a_score_i,timeI):

	gd_i = h_score_i - a_score_i
	myobj = robjects.r('''
	require(leaps)
	require(ggplot2)
	library(nnet)
	library(glmnet)
	set.seed(3)

	allData = read.csv('Project_AnalyticalDataset_LogReg.csv')
	attach(allData)

	#convert outcome to type factor and relevel
	allData$outcomeF <- factor(allData$outcome)
	allData$out <- relevel(allData$outcomeF, ref = "1")
	#remove irrelevant columns
	dropNames <- c("home_team","away_team","match_id","match_date","country",
	               "stadium","outcomeF","outcome","home_score","away_score",
	               "home_score_gd", "home_percent_win", "home_percent_points",
	               "away_score_gd",	"away_percent_win",	"away_percent_points",
	               "score_diff",	"percent_win_diff",	"percent_points_diff")

	allData <- allData[,!(names(allData)) %in% dropNames]

	# ------------------ Begin self defined functions ------------------

	#gets user input and returns a vector of independent variables
	#to be passed into predProbs()
	getUserInput <- function(hYC_i, h_Subs_i, h_RC_i,
	                         aYC_i, a_Subs_i, a_RC_i,
	                         gd_i,
	                         h_score_gd_i, h_pct_win_i, h_pct_pts_i,
	                         a_score_gd_i, a_pct_win_i, a_pct_pts_i){
	  #add 1 to account for intercept
	  return(c(1,hYC_i, h_Subs_i, h_RC_i,
	           aYC_i, a_Subs_i, a_RC_i,
	           gd_i,
	           h_score_gd_i, h_pct_win_i, h_pct_pts_i,
	           a_score_gd_i, a_pct_win_i, a_pct_pts_i))
	}

	#takes the glm model returned by glmnet and returns the coef
	#of of the i^th model
	retCoef <- function(fit1,index){
	  coefMatrix <- matrix(coef(fit1)[[index]])
	  return(as.vector(coefMatrix))
	}

	#takes the vector of coeff returned by retCoef
	#and multiplies with user parameters
	linSum <- function(betaI,userParams){
	  xI <- as.vector(userParams)
	  wI <- betaI %*% xI
	  return(wI)
	}

	predProbs <- function(fit1,userInput){
	  #win
	  coef1 <- retCoef(fit1,1)
	  w1 <- linSum(coef1,userInput)
	  #lose
	  coef2 <- retCoef(fit1,2)
	  w2 <- linSum(coef2,userInput)
	  #draw
	  coef3 <- retCoef(fit1,3)
	  w3 <- linSum(coef3,userInput)
	  
	  denom <- exp(w1) + exp(w2) + exp(w3)
	  pred1 <- exp(w1)/denom
	  pred2 <- exp(w2)/denom
	  pred3 <- exp(w3)/denom
	  #return probabilities for win,lose,draw
	  return(c(pred1,pred2,pred3))
	}



	labelMaker <- function(timeI){
	  targetTime = timeI
	  hBase = "home"
	  aBase = "away"
	  hYC <- paste(hBase,"yellowCards",targetTime,sep = "_")
	  hSubs <- paste(hBase,"subIns",targetTime,sep="_")
	  hRC <- paste(hBase,"redCards",targetTime,sep="_")
	  hGdiff <- "home_score_gd_n"
	  hPercWin <- "home_percent_win_n"
	  hPercPts <- "home_percent_points_n"
	  
	  aYC <- paste(aBase,"yellowCards",targetTime,sep = "_")
	  aSubs <- paste(aBase,"subIns",targetTime,sep="_")
	  aRC <- paste(aBase,"redCards",targetTime,sep="_")
	  
	  aGdiff <- "away_score_gd_n"
	  aPercWin <- "away_percent_win_n"
	  aPercPts <- "away_percent_points_n"
	  
	  gDiff <- paste("GD",targetTime,sep="_")
	  
	  targetNames <- c(hYC,hSubs,hRC,hGdiff,hPercWin,hPercPts,
	                   aYC,aSubs,aRC,aGdiff,aPercWin,aPercPts,
	                   gDiff,"out")#,sDiff,pWinDiff,pPtsDiff)
	  return(targetNames)
	}


	#user inputs timeI, returns glmnet multinomial logit model with
	#test accuracy determined using our own test data
	buildModel <- function(timeI,allData1){
	  # timeI = 5 
	  # allData1 = allData
	  targetNames <- labelMaker(timeI)
	  allDataI <- allData1[,(names(allData1)) %in% targetNames]
	    
	  y1 <- allDataI[,"out"]
	  dropNames6 <- c("out")
	  allDataI <- allDataI[,!(names(allDataI)) %in% dropNames6]
	  x1 <- as.matrix(allDataI)
	  
	  trainIndicEnd <- floor(0.75*nrow(allDataI))
	  trainIndic <- sample(nrow(allDataI),trainIndicEnd)
	  testIndic <- -trainIndic
	  trainDataX <- x1[trainIndic,]
	  trainDataY <- y1[trainIndic]
	  testDataX <- x1[testIndic,]
	  testDataY <- y1[testIndic]
	  
	  #Cross validation to get best lambda
	  logreg2 <- cv.glmnet(x = trainDataX,y = trainDataY,family="multinomial",
	                       type.multinomial = "grouped", parallel=TRUE,
	                       type.measure="class",nfolds=3)
	  
	  bestLambda <- logreg2$lambda.min
	  
	  #retrain model using best lambda
	  logreg3 <- glmnet(x = trainDataX,y = trainDataY,family = "multinomial",
	                    lambda = bestLambda, type.multinomial = "grouped")
	  
	  #predict win/lose/draw
	  pred3 <- predict(logreg3,newx = testDataX,type = "class")
	  #calculate accuracy
	  hitClass <- ifelse(pred3 == testDataY,1,0)
	  accuracy <- sum(hitClass)/length(testDataY)
	  # print(accuracy)
	  logreg3["testAccuracy"] <- accuracy
	  return(logreg3)
	}

	#plots a graph of winning probabilities against time if
	#user input remains the same for the game
	churnGraph <- function(allData1,userInput,outFilename){
	  winProb <- c()
	  loseProb <- c()
	  drawProb <- c()
	  allTimes <- seq(5,85,5)
	  for (timeI in allTimes){
	    print(timeI)
	    modelI <- buildModel(timeI,allData1)
	    predResults <- predProbs(modelI,userInput)
	    print(predResults)
	    winProb <- c(winProb,predResults[1])
	    loseProb <- c(loseProb,predResults[2])
	    drawProb <- c(drawProb,predResults[3])
	  }
	  print('graphing...')
	  #concat into dataframe before calling ggplot
	  df1 <- data.frame(allTimes,winProb,loseProb,drawProb)
	  return(df1)
	}


	## ------------------ End self defined functions ------------------

	''')
	
	#create dynamic logit graph .png name
	currDateTime = getDateTimeStr()
	outfileName1 = "logit_probs_%s.png" % (currDateTime) 
	graphOutfilename = "static/%s" % (outfileName1)
	
	#get R functions from global environment
	getUserInputPy = robjects.globalenv['getUserInput']
	allDataPy = robjects.globalenv['allData']
	churnGraphPy = robjects.globalenv['churnGraph']
	buildModelPy = robjects.globalenv['buildModel']

	#prepare user input
	userInput1 = getUserInputPy(hYC_i, h_Subs_i, h_RC_i,
	                         aYC_i, a_Subs_i, a_RC_i,
	                         gd_i,
	                         h_score_gd_i, h_pct_win_i, h_pct_pts_i,
	                         a_score_gd_i, a_pct_win_i, a_pct_pts_i)


	output1 = churnGraphPy(allDataPy,userInput1,graphOutfilename)

	probListObj = output1[1:len(output1)]
	probColObj = output1[0]

	print('------ extracting ------- \n')

	probList = extractRvalues(probListObj)


	print('------ getting columns ------ \n')
	probCol = list()
	for i in probColObj:
		probCol.append(int(i))

	print('-------- converting to dataframe ---------\n ')
	probDF = pd.DataFrame(probCol,columns=["TIME"])
	probRes = pd.DataFrame(probList).transpose()
	probRes.columns =["P_WIN","P_LOSE","P_DRAW"]

	print('--------- concat --------- \n')
	probResult = pd.concat([probDF,probRes],axis=1)
	print(probResult)

	#plot probResult
	plt.plot(probResult.TIME,probResult.P_WIN,'g-s',linewidth=2.0,label="Home wins")
	plt.plot(probResult.TIME,probResult.P_LOSE,'r-o',linewidth=2.0,label = "Away wins")
	plt.plot(probResult.TIME,probResult.P_DRAW,'b-^',linewidth=2.0,label = "Draw")
	plt.ylabel('Probability',fontsize = 20)
	plt.xlabel('Time',fontsize = 20)
	plt.title('Probability against Time',fontsize = 20)
	plt.legend(['P(Home wins)','P(Away wins)','P(Draw)'],loc="upper left")
	plt.ylim([0,1])
	plt.savefig(graphOutfilename)
	plt.close()

	
	last_row = probResult[probResult["TIME"]==timeI][["P_WIN", "P_LOSE", "P_DRAW"]]
	last_row["max"] = last_row.idxmax(axis=1)
	lr_final_pred = last_row["max"].values[0]
	lr_final_prob = last_row[last_row["max"]].values[0][0]

	#Converting prediction into a number which will be used to determine the output for this model
	if lr_final_pred == "P_WIN":
		lr_final_result = 1
	elif lr_final_pred == "P_LOSE":
		lr_final_result = 0
	else:
		lr_final_result = 2

	return lr_final_result, lr_final_prob, outfileName1

#Function runs k-nearest neighbors (KNN)
def run_knn(userTime, homeTeam, awayTeam, homeScore, awayScore, homeYC, awayYC, homeRC, awayRC, homeSubs, awaySubs, knn_score_gd, knn_pct_win, knn_pct_pts):
	KNN_model = '''
	function(homeTeam, awayTeam, homeScore, awayScore, homeYC, awayYC, homeRC, awayRC, homeSubs, awaySubs, knn_score_gd, knn_pct_win, knn_pct_pts){
	
	soccerdb<-read.csv("Project_AnalyticalDataset_KNN.csv",na.strings="NA", header = TRUE)

	# install.packages("class")
	library(class)
	
	time<-list(5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85)
	results<-list()
	m2<-60
	data <- as.data.frame(data.frame(prediction= numeric(0), prob= numeric(0)))
	# this runs the model for each of the 5 minute interval and stores the result in vector
	for (i in 1:17)
	{
	  n<-24-i+1
	  #Getting the relevant columns
	  temp<-soccerdb[,c(2,3,4,5,n,n+38,n+57,n+133,n+171,n+190,n+266,299:301,292)]
	  TrainData <- temp[1:9122,] # select all these rows
	  TestData<-TrainData[1,]
	  # Assigning the user inputs to the test data
	  TestData$match_id='0000000'
	  TestData$home_team=homeTeam
	  TestData$away_team=awayTeam
	  TestData$match_date='28/08/2011'
	  TestData[,5]=homeYC
	  TestData[,6]=homeSubs
	  TestData[,7]=homeRC
	  TestData[,8]=awayYC
	  TestData[,9]=awaySubs
	  TestData[,10]=awayRC
	  TestData[,11]=homeScore-awayScore
	  TestData[,12]=knn_score_gd
	  TestData[,13]=knn_pct_win
	  TestData[,14]=knn_pct_pts
	  TestData$outcome_test=knn(TrainData[,c(5,6,7,8,9,10,11,12,13,14)], TestData[,c(5,6,7,8,9,10,11,12,13,14)], TrainData$outcome, k = 75, prob=TRUE)
	  outcomedata<-data.frame(TestData, prediction=TestData$outcome_test, prob=attr(TestData$outcome_test, "prob"))
  	  data=rbind(data,outcomedata[,c(17,18)])
	}
	# Converting the factor to a numeric quantity to be able to plot the values
	data$prediction<-as.numeric(as.character(data$prediction))
	for (i in 1:17){
	data$Outcome<-ifelse(data$prediction==0,'Away team wins',ifelse(data$prediction==1, 'Home team wins', 'Draw'))
	}
	return(data)
	}
	'''
	knn_mod = robjects.r(KNN_model)
	knn_result = knn_mod(homeTeam, awayTeam, homeScore, awayScore, homeYC, awayYC, homeRC, awayRC, homeSubs, awaySubs, knn_score_gd, knn_pct_win, knn_pct_pts)
	knnIdx = int(userTime/5) -1
	knn_final_pred = knn_result[0][knnIdx]
	knn_final_prob = knn_result[1][knnIdx]
	
	#Converting R dataframe to pandas dataframe
	knn_df = com.convert_robj(knn_result)
	knn_df2 = pd.DataFrame()
	knn_df2["Probability"] = np.round(knn_df["prob"],2)
	knn_df2["Outcome"] = knn_df["Outcome"]
	knn_df2["Time"] = np.arange(5,90,5)
	knn_df2 = knn_df2.set_index("Time")

	#Transposing dataframe which will be displayed on the results page of our app
	knn_df3 = knn_df2.transpose()

	return knn_final_pred, knn_final_prob, knn_df3

# Extracts data from csv file and returns info
def get_data(name_file):    
    if os.path.exists(name_file):        
        data = pd.read_csv(name_file, index_col=0, encoding='latin-1')
    else:
        print("-- file not found")
        exit("-- Unable to open file")
    return data

# Creates Decision Tree and returns it
def train_tree(data, features):
    
    # Creates train(70%) and test(30%) data
    train, test = train_test_split(data, test_size = 0.3)
    x_train = train[features]
    y_train = train[['outcome']]
    x_test = test[features]
    y_test = test[['outcome']]
    
    # Loops until it finds the best decision tree
    n=0
    incumbent=0
    dif=1
    while dif>0.025:
        n+=1
        dt = DecisionTreeClassifier(max_depth=n,criterion="entropy")
        dt.fit(x_train, y_train)        
        dif = abs(dt.score(x_train,y_train) - incumbent)
        incumbent = dt.score(x_train,y_train)
    n+=-1
    dt = DecisionTreeClassifier(max_depth=n,criterion="entropy")
    dt.fit(x_train, y_train)
    predicted = cross_val_predict(dt, data[features], data['outcome'])
    accuracy = accuracy_score(data[['outcome']], predicted)
         
    return [dt, n, accuracy]

# Creates dot and png output for the tree. There is a problem with my laptop and I can't print the tree on the screen.
#def print_tree(dt, features, minute):
    #dotfile = tree.export_graphviz(dt.tree_, out_file='decision_tree.dot', feature_names=features)     
    #dot_data = tree.export_graphviz(dt.tree_, out_file=None, feature_names=features) 
    #graph = pydotplus.graph_from_dot_data(dot_data)     
                    
    #parent = os.path.join(os.getcwd(), "..")    
    #name_path = parent+'\\static\\'
    #graph.write_png(name_path + "decision_tree_" + str(minute) + ".png")

def get_results(tree, feature_names, test, n):
    
    result = tree.tree_.value    
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
    # get ids of child nodes
    idx = np.argwhere(left == -1)[:,0] 

    def recurse(left, right, child, lineage=None):
        if lineage is None:
            lineage = [child]
        if child in left:
            parent = np.where(left == child)[0].item()
            split = 'l'
        else:
            parent = np.where(right == child)[0].item()
            split = 'r'

        lineage.append((parent, split, threshold[parent], features[parent]))

        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)
    
    for child in idx:
        cont = 0
        for node in recurse(left, right, child):
            cont+=1
            if cont <= n:
                if node[1] == 'l':
                    if test[node[3]] <= node[2]:
                        if cont == n:
                            home = result[child][0][1]
                            away = result[child][0][0]
                            tie = result[child][0][2]
                            total = home+away+tie
                    else:
                        break
                elif node[1] == 'r':
                    if test[node[3]] > node[2]:
                        if cont == n:
                            home = result[child][0][1]
                            away = result[child][0][0]
                            tie = result[child][0][2]
                            total = home+away+tie
                    else:
                        break
                        
    return [home/total*100, away/total*100, tie/total*100, total]

def get_variables(minute):
    minute = str(int(round(minute/5)*5))
    var = ["GD_" + minute,        
        "home_yellowCards_" + minute, 
        "away_yellowCards_" + minute, 
        "home_redCards_" + minute, 
        "away_redCards_" + minute, 
        "home_subOuts_" + minute, 
        "away_subOuts_" + minute,
        "diff_level"]
    return var

def get_test(minute, dicc):
    minute = str(int(round(minute/5)*5))
    dicc["GD_" + minute] = dicc.pop("GD_")
    dicc["home_yellowCards_" + minute] = dicc.pop("home_yellowCards_" )
    dicc["away_yellowCards_" + minute] = dicc.pop("away_yellowCards_")
    dicc["home_redCards_" + minute] = dicc.pop("home_redCards_")
    dicc["away_redCards_" + minute] = dicc.pop("away_redCards_")
    dicc["home_subOuts_" + minute] = dicc.pop("home_subOuts_")
    dicc["away_subOuts_" + minute] = dicc.pop("away_subOuts_")
    return dicc

#Creating result plot
def create_plot(table_results, name_file):
    
    df = pd.DataFrame(table_results, columns=['Home', 'Away', 'Tie', 'Total', 'Minute', 'Accuracy'])

    plt.plot(df.Minute, df.Home,'g-^', label='Home')
    plt.plot(df.Minute, df.Away,'r-o', label='Away')
    plt.plot(df.Minute, df.Tie,'b-s', label='Tie')
    plt.ylabel('Probability')
    plt.xlabel('Time')
    plt.title('Decision Tree Prediction')
    matplotlib.rcParams.update({'font.size': 18})
    plt.legend(loc='best')
    plt.yticks(range(0, 100, 10), [str(x) + "%" for x in range(0, 100, 10)], fontsize=14) 

    nameI = 'static/%s' %(name_file)
    plt.savefig(nameI)
    plt.close()

def get_performance(home, away):
    data = pd.read_csv('teams_performance.csv', index_col=0, encoding='latin-1')
    home_perf = data.loc[data['Team'] == home].iloc[0]['Home']
    away_perf = data.loc[data['Team'] == away].iloc[0]['Away']    
    dif_perf = home_perf - away_perf
    return dif_perf

#Function runs decision tree model
def run_dr(homeScore,awayScore,hyc,ayc,hrc,arc,hsub,asub,hteam,ateam,minute):
    
    gdif = int(homeScore - awayScore)
    minute_table = int(round(minute/5))
    data = get_data("Project_AnalyticalDataset_DR.csv")
    results = []
    ldif = get_performance(hteam,ateam)
    
    for time in range(0,17):
        
        input_test = {"GD_": gdif,"home_yellowCards_": hyc,"away_yellowCards_": ayc, "home_redCards_": hrc,"away_redCards_": arc, "home_subOuts_": hsub, "away_subOuts_": asub, "diff_level": ldif}
        
        variables = get_variables(int((time+1)*5))        
        input_format = get_test(int((time+1)*5), input_test)
        [tree, n, accuracy] = train_tree(data, variables)
        aux = get_results(tree, variables, input_test, n)        
        aux.extend([int((time+1)*5), accuracy*100])
        results.append(aux)

    result_game = [1, 0, 2]       
    index = np.argmax(results[minute_table][0:3])
    percent = results[minute_table][index]
    
    #Creating dynamic name for the decision tree graph
    dr_fig = 'decision_tree_' +str(gdif)+str(hyc)+str(ayc)+str(hrc)+str(arc)+str(hsub)+str(asub)+str(int(ldif))+'.png'
    create_plot(results, dr_fig)

    return result_game[index], percent/100, dr_fig

#Rendering homepage
@app.route('/')
def main():
	session['data_loaded'] = True
	return render_template('home.html')

#Input page - User defines game parameters and choses one of the three models
@app.route('/input',methods=['POST','GET'])
def input():
	infoForm = dropForm()
	if infoForm.validate_on_submit():
		userTime = int(request.form.get('scoreTime'))
		homeTeam = str(request.form.get('homeTeam')).lower().strip()
		awayTeam = str(request.form.get('awayTeam')).lower().strip()
		homeScore = int(request.form.get('homeScore'))
		awayScore = int(request.form.get('awayScore'))
		homeYC = int(request.form.get('homeYC'))
		awayYC = int(request.form.get('awayYC'))
		homeRC = int(request.form.get('homeRC'))
		awayRC = int(request.form.get('awayRC'))
		homeSubs = int(request.form.get('homeSubs'))
		awaySubs = int(request.form.get('awaySubs'))
		model_type = str(request.form.get('model_type'))
		
		#Extractng home and away metrics based on user-defined parameters
		hMetrics = getMetrics(homeTeam, home=True)
		aMetrics = getMetrics(awayTeam, home=False)

		h_score_gd = hMetrics[0]
		h_pct_win = hMetrics[1]
		h_pct_pts = hMetrics[2]

		a_score_gd = aMetrics[0]
		a_pct_win = aMetrics[1]
		a_pct_pts = aMetrics[2]

		#Generating metrics for the KNN model
		knn_score_gd = h_score_gd - a_score_gd
		knn_pct_win = h_pct_win - a_pct_win
		knn_pct_pts = h_pct_pts - a_pct_pts

		#Running a model based on user preference
		if model_type == 'Logistic_Regression':
			mod_out1, mod_out2, mod_out3 = run_log_reg(homeTeam, awayTeam, homeYC,homeSubs,homeRC,h_score_gd,h_pct_win,h_pct_pts,awayYC,awaySubs,awayRC,a_score_gd,a_pct_pts,a_pct_win,homeScore,awayScore,userTime)
		elif model_type == 'Decision_Trees':
			mod_out1, mod_out2, mod_out3 = run_dr(homeScore, awayScore, homeYC, awayYC, homeRC, awayRC, homeSubs, awaySubs, homeTeam.lower(), awayTeam.lower(), userTime)
		else:
			mod_out1, mod_out2, mod_out3 = run_knn(userTime, homeTeam, awayTeam, homeScore, awayScore, homeYC, awayYC, homeRC, awayRC, homeSubs, awaySubs, knn_score_gd, knn_pct_win, knn_pct_pts)

		mod_out2 = 100*round(mod_out2,2)

		#Displaying outputs for testing
		print("\n------------------ Outcome --------------------------\n")
		print(mod_out1)
		print("\n------------------ Probability --------------------------\n")
		print(mod_out2)
		print("\n------------------ Graph/Dataframe --------------------------\n")
		print(mod_out3)

		#Rendering the dataframe as an html table
		if model_type == 'KNN':
			mod_out3 = mod_out3.to_html()

		return render_template("result.html", homeTeam = homeTeam.upper(), awayTeam = awayTeam.upper(), userTime = userTime, \
			mod_type = model_type, mod_out1 = mod_out1, mod_out2 = mod_out2, mod_out3 = mod_out3)

	else:
		print('false')
		print(infoForm.errors) #Printing errors (if any)

	return render_template("inputparams.html",form=infoForm)

#Running the app in debug mode 
if __name__ == "__main__":
	app.run(debug=True)
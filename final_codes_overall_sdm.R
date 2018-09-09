#The species occurrence data for "Mikania micrantha Kunth" was downloaded from GBIF.
#############################
#Occurrence points for South America
#############################
library(dismo)
mic_gbif = gbif (genus = "Mikania", species="micrantha Kunth", geo = TRUE)
mic_gbif = mic_gbif[, c("species", "lon", "lat")] 
#The NA values in the occurrence data was removed.
mic_gbif = subset(mic_gbif, !is.na(lon) &	!is.na(lat) & !is.na(species)) 
#The data that was recorded multiple times was removed.
unique(mic_gbif$species)
dim(mic_gbif)
#The occurrence data was plotted on a world map.
library(maptools) 
data("wrld_simpl") 
xlim = c(min(mic_gbif$lon), max(mic_gbif$lon)) 
ylim = c(min(mic_gbif$lat), max(mic_gbif$lat))
plot(wrld_simpl, xlim = xlim, ylim = ylim, axes = TRUE, col = "light yellow") 
box()
points(mic_gbif$lon, mic_gbif$lat, col = "red", pch = 20, cex = 0.75)
mic_gbif_america = subset(mic_gbif, lon< (-30))
plot(wrld_simpl, xlim = c(-110, -34), 	ylim = 	c(-50, 30), axes = TRUE, col = "light yellow") 
points(mic_gbif_america$lon, 	mic_gbif_america$lat, col = "red",lwd= 2)
write.csv(mic_gbif_america, 	"mic_gbif_america.csv", row.names = 	FALSE)
########################
#Occurrence points for India
#####################
india = read.csv("india_occur_final.csv")
mic_gbif_india = india[,2:3]
names(mic_gbif_india) = c("lon", "lat")
mic_gbif_india = write.csv(mic_gbif_india, "mic_gbif_india.csv", row.names = FALSE)
mic_gbif_india = read.csv("mic_gbif_india.csv")
coordinates(mic_gbif_india) = ~lon+lat
crs(mic_gbif_india) = crs(wrld_simpl)
class(mic_gbif_india)
#The bio-climatic data were downloaded from WORLDCLIM as stack of raster layers. Each layer consisted of one variable.  
library(raster)
bioclim_data=getData(name="worldclim", download=TRUE, var="bio", res=2.5, path="H:/My Project/R-Codes/rds files")
class(bioclim_data) 
writeRaster(bioclim_data, filename ="bioclim_data.grd ", overwrite = T) 
bioclim_data = brick("bioclim_data.grd")
bioclim_data_world = crop(bioclim_data, wrld_simpl)
writeRaster(bioclim_data_world, filename ="bioclim_data_world.grd ", overwrite = T) 
bioclim_data_wrld = brick("bioclim_data_world.grd ")
#In order to mask the bio-climatic data we need the extent to which it has to be masked. The region of interests were downloaded from GADM as SpatialPolygonsDataFrame and the climatic data was cropped to its extent. 
###################
#Region of South America
###################
Mexico = readRDS("MEX_adm0.rds") # A part of North America
crs(Mexico) = crs(wrld_simpl)

Venezuela = readRDS("VEN_adm0.rds")
crs(Venezuela) = crs(wrld_simpl)

Brazil = readRDS("BRA_adm0.rds")
crs(Brazil) = crs(wrld_simpl)

Columbia = readRDS("COL_adm0.rds")
crs(Columbia) = crs(wrld_simpl)

Peru = readRDS("PER_adm0.rds")
crs(Peru) = crs(wrld_simpl)

Bolivia = readRDS("BOL_adm0.rds")
crs(Bolivia) = crs(wrld_simpl)

Chile = readRDS("CHL_adm0.rds")
crs(Chile) = crs(wrld_simpl)

Argentina = readRDS("ARG_adm0.rds")
crs(Argentina) = crs(wrld_simpl)

Paraguay = readRDS("PRY_adm0.rds")
crs(Paraguay) = crs(wrld_simpl)

Uruguay = readRDS("URY_adm0.rds")
crs(Uruguay) = crs(wrld_simpl)

Guyana = readRDS("GUY_adm0.rds")
crs(Guyana) = crs(wrld_simpl)

Suriname = readRDS("SUR_adm0.rds")
crs(Suriname) = crs(wrld_simpl)

Ecuador = readRDS("ECU_adm0.rds")
crs(Ecuador) = crs(wrld_simpl)

French_Guiana = readRDS("GUF_adm0.rds")
crs(French_Guiana) = crs(wrld_simpl) 

#Central America
Guatemala = readRDS("GTM_adm0.rds")
crs(Guatemala) = crs(wrld_simpl)

Nicaragua = readRDS("NIC_adm0.rds")
crs(Nicaragua) = crs(wrld_simpl)  

Costa_Rica = readRDS("CRI_adm0.rds")
crs(Costa_Rica) = crs(wrld_simpl)  

Panama = readRDS("PAN_adm0.rds")
crs(Panama) = crs(wrld_simpl)

Honduras = readRDS("HND_adm0.rds")
crs(Honduras) = crs(wrld_simpl)

Belize = readRDS("BLZ_adm0.rds")
crs(Belize) = crs(wrld_simpl)

#Islands near South America in the Carribean sea
Cuba = readRDS("CUB_adm0.rds")
crs(Cuba) = crs(wrld_simpl)

Haiti = readRDS("HTI_adm0.rds")
crs(Haiti) = crs(wrld_simpl)

Jamaica = readRDS("JAM_adm0.rds")
crs(Jamaica) = crs(wrld_simpl)

Bahamas = readRDS("BHS_adm0.rds")
crs(Bahamas) = crs(wrld_simpl)

Dominician_Republic = readRDS("DOM_adm0.rds")
crs(Dominician_Republic) = crs(wrld_simpl)

Trinidad_Tobago = readRDS("TTO_adm0.rds")
crs(Trinidad_Tobago) = crs(wrld_simpl)

Puerto_Rico = readRDS("PRI_adm0.rds")
crs(Puerto_Rico) = crs(wrld_simpl)

Faulkland = readRDS("FLK_adm0.rds")
crs(Faulkland) = crs(wrld_simpl)

Dominica = readRDS("DMA_adm0.rds")
crs(Dominica) = crs(wrld_simpl)

El_Salvador = readRDS("SLV_adm0.rds")
crs(El_Salvador) = crs(wrld_simpl)

Grenada = readRDS("GRD_adm0.rds")
crs(Grenada) = crs(wrld_simpl)

Guadeloupe = readRDS("GLP_adm0.rds")
crs(Guadeloupe) = crs(wrld_simpl)

Martinique = readRDS("MTQ_adm0.rds")
crs(Martinique) = crs(wrld_simpl)

Saint_Lucia = readRDS("LCA_adm0.rds")
crs(Saint_Lucia) = crs(wrld_simpl)

south_america = rbind(Mexico, Venezuela, Brazil, Columbia, Peru, Bolivia, Chile, Argentina, Paraguay, Uruguay, Guyana, Suriname, Ecuador, French_Guiana, Guatemala, Nicaragua, Costa_Rica, Panama, Honduras, Cuba, Haiti, Jamaica, Bahamas, Dominician_Republic, Trinidad_Tobago, Puerto_Rico, Faulkland, Belize, Dominica, El_Salvador, Grenada, Guadeloupe, Martinique, Saint_Lucia)
saveRDS(south_america, "south_america.rds")
plot(south_america)
box()
class(south_america)

#Masking climatic variables for the two regions:
############
#South America
############
bio_america = mask(bioclim_data_world, south_america)
writeRaster(bio_america, filename ="bio_america.grd ", overwrite = T) 
bio_america = brick("bio_america.grd")
class(bio_america)
plot(bio_america, 1:9)
plot(bio_america, 10:18)
############
#Region of India
############
#india = getData(name = "GADM", download = TRUE, country = "IND", level = 0)

bioclim_data_india = mask(bioclim_data_world, india)
writeRaster(bioclim_data_india, "bioclim_data_india.grd", overwrite = TRUE)
bioclim_data_india = brick("bioclim_data_india.grd")
class(bioclim_data_india)
plot(bioclim_data_india, 1:9)
plot(bioclim_data_india, 10:18)

#Extracted the data from raster stack of environmental variables for the longitude-latitude data of the species occurrence and remove the missing values.
predictors_values_america = extract(bio_america, mic_gbif_america[, c("lon", "lat")]) 
ind = complete.cases(as.data.frame(predictors_values_america)) 
predictors_values_america = as.data.frame(predictors_values_america[ind,])
sum(ind)

mic_gbif_india = read.csv("mic_gbif_india.csv")
predictors_values_india = extract(bioclim_data_india, mic_gbif_india[, c("lon", "lat")]) # class-- matrix
ind = complete.cases(as.data.frame(predictors_values_india)) 
predictors_values_india = as.data.frame(predictors_values_india[ind,])
sum(ind)
write.csv(predictors_values_india, "predictors_values_india.csv", row.names = F)
predictors_values_india = read.csv("predictors_values_india.csv")
#Removing the selection bias for the region of South America only.
dups = duplicated(mic_gbif_america[,c('lon', 'lat')])
sum(dups)
mic_gbif_america = mic_gbif_america[!dups,]
#Removing the geographic bias : The points that fall in the ocean were removed and only one occurrence point form each grid was selected to overcome the geographic bias.
coordinates(mic_gbif_america) = ~lon+lat
crs(mic_gbif_america ) = crs(wrld_simpl)
class(mic_gbif_america)
ovr = over(mic_gbif_america, wrld_simpl)
class(ovr)
head(ovr)
country = ovr$NAME
sum(is.na(country))
is_ocean = which(is.na(country))
lonlat_ocean = mic_gbif_america@coords[is_ocean,]
mic_gbif_america@coords = mic_gbif_america@coords[-is_ocean,]
plot(mic_gbif_america, col = "green", cex= 0.5, pch=16)
plot(wrld_simpl, add=TRUE, lwd = 2)
points(lonlat_ocean, col="red", cex = 2, pch=3, lwd= 2)
box()
r = raster(mic_gbif_america)
res(r) = 0.3
r = extend(r, extent(r)+0.2)
occ_one = gridSample(mic_gbif_america, r, n=1)
class(occ_one); dim(occ_one)
p = rasterToPolygons(r)
plot(p, border = 'gray')
points(mic_gbif_america)
points(occ_one, cex = 1, col = 'red',  pch = 'x')
dim(occ_one)
write.csv(occ_one, "mic_gbif_america.csv", row.names = FALSE)
file = paste(system.file(package = "dismo"), '/cleaned/mic_gbif_america.csv', sep = ' ')
#Now a minimum convex polygon is created so that it covers all the occurrence points.
##############
#For South America
##############
library(adehabitatHR)
Convex_poly = mcp(mic_gbif_america, percent = 100)
plot(wrld_simpl, xlim = c(-110, -34),  ylim =  c(-50, 30),  axes = TRUE, col = "light yellow") 
plot(Convex_poly,  add = T)
class(Convex_poly)
library(rgeos)
Min_convex_poly = gIntersection(wrld_simpl, Convex_poly)
plot(Min_convex_poly, lwd = 1)
points(mic_gbif_america, col = "blue", cex = 0.9, pch = 20)
Min_convex_poly_extended = gBuffer(Min_convex_poly, byid=FALSE, id=NULL, width=1.0, quadsegs=5, capStyle="ROUND",
                                   joinStyle="ROUND", mitreLimit=1.0)
plot(Min_convex_poly_extended, lwd = 1)
points(mic_gbif_america, col = "red", cex = 0.9, pch = 20)
bioclim_data_mcp = mask(crop(bio_america, Min_convex_poly_extended), Min_convex_poly_extended)
writeRaster(bioclim_data_mcp, "bioclim_data_mcp.grd", overwrite = T)
bioclim_data_mcp = brick("bioclim_data_mcp.grd")
############
#Region of India
############
Convex_poly_india = mcp(mic_gbif_india, percent = 100)
plot(wrld_simpl, xlim = c(65, 100), 	ylim = 	c(5, 45), axes = TRUE, col = "light yellow") 
plot(Convex_poly_india, add = T)
class(Convex_poly_india)
library(rgeos)
Min_convex_poly_india = gIntersection(wrld_simpl, Convex_poly_india)
plot(Min_convex_poly_india, lwd = 1)
points(mic_gbif_india, col = "blue", cex = 0.9, pch = 20)
Min_convex_poly_extended_india= gBuffer(Min_convex_poly_india, byid=FALSE, id=NULL, width=1.0, quadsegs=5, capStyle="ROUND",
                                        joinStyle="ROUND", mitreLimit=1.0)
plot(Min_convex_poly_extended_india, lwd = 1)
points(mic_gbif_india, col = "red", cex = 0.9, pch = 20)
bioclim_data_india_mcp = mask(crop(bioclim_data_india, Min_convex_poly_extended_india), Min_convex_poly_extended_india)
writeRaster(bioclim_data_india_mcp, "bioclim_data_india_mcp.grd", overwrite = T)
bioclim_data_india_mcp = brick("bioclim_data_india_mcp.grd")

#Generating background points :
##############
#For South America
##############
library(dismo)
background = randomPoints(bioclim_data_mcp, 874)
pseudo_absence_values = extract(bioclim_data_mcp, background)
write.csv("pseudo_absence_values.csv")
pseudo_absence_values = read.csv("pseudo_absence_values.csv")
############
#Region of India
############
background_india = randomPoints(bioclim_data_india_mcp, 105)
pseudo_absence_values = extract(bioclim_data_india_mcp, background_india)
write.csv(pseudo_absence_values, "pseudo_absence_values.csv", row.names = F)
pseudo_absence_values = read.csv("pseudo_absence_values.csv")
#The following data is generated for obtaining the combined occurrence points from South America and India:
mic_gbif_america_india = rbind(mic_gbif_america, mic_gbif_india)
write.csv(mic_gbif_america_india, "mic_gbif_america_india.csv", row.names = F)
predictors_values_america_india = rbind(predictors_values_america, predictors_values_india)
write.csv(predictors_values_america_india, "predictors_values_america_india.csv", row.names = F)
#The codes below is used to create logistic, ridge and lasso regression models.  Also the test error, sensitivity, F measure,  Area under curve (AUC) is calculated using the following codes. The variable selection is done while running ridge and lasso regression models.
library(raster)
library(dismo)
library(glmnet)
library(car)
library(stats)
library(pROC)
bioclim_data_america = brick("bioclim_data_america.grd")
bioclim_data_india = brick("bioclim_data_india.grd")
bioclim_data_india_mcp = brick("bioclim_data_india_mcp.grd")
bioclim_data_mcp = brick("bioclim_data_mcp.grd")
mic_gbif_america  = read.csv("mic_gbif_america.csv", header = TRUE)
predictors_values_america  = read.csv("predictors_values_america.csv", header = TRUE)
predictors_values_america_india = read.csv("predictors_values_america_india.csv", header = TRUE)
predictors_values_india = read.csv("predictors_values_india.csv", header = TRUE)
n = 10     # Number of background to be chosen
run = 30   # Number of time the model should be run for average test error
test.error = matrix(NA, nrow = run, ncol = n)
f.measure = matrix(NA, nrow = run, ncol = n)
s = matrix(NA, nrow = run, ncol = n)
auc_logistic = matrix(NA, nrow = run, ncol = n)
auc_ridge = matrix(NA, nrow = run, ncol = n)
val_logistic = rep(FALSE,18)
selected_logistic = rep(FALSE,18)
val_ridge = rep(FALSE,19)
selected_ridge = rep(FALSE,19)
background_type = "sa_mcp"  #sa_mcp, sa_in, sa_in_mcp
# sa = South America; 
# sa_mcp = South America minimum convex polygon
# sa_in = South America and India
# sa_in_mcp = South America mcp and India mcp 
#########################
#Run of the logistic regression  
#########################
for(i in 1:n)
{
  if(background_type == "sa"){
    background = randomPoints(bioclim_data_america, nrow(predictors_values_america))
    pseudo_absence_values = extract(bioclim_data_america, background)
    y = as.factor(c(rep(1, nrow(predictors_values_america)), rep(0, nrow(pseudo_absence_values))))
    sdm_data_america = cbind(y, rbind(predictors_values_america, pseudo_absence_values))
  }
  
  if(background_type == "sa_mcp"){
    background = randomPoints(bioclim_data_mcp, nrow(predictors_values_america))
    pseudo_absence_values = extract(bioclim_data_mcp, background)
    y = as.factor(c(rep(1, nrow(predictors_values_america)), rep(0, nrow(pseudo_absence_values))))
    sdm_data_america = cbind(y, rbind(predictors_values_america, pseudo_absence_values))
  }
  
  if(background_type == "sa_in_mcp"){
    background.america = randomPoints(bioclim_data_mcp, nrow(predictors_values_america))
    background.india = randomPoints(bioclim_data_india_mcp,nrow(predictors_values_india))
    pseudo_absence_values.america = extract(bioclim_data_mcp, background.america)
    pseudo_absence_values.india = extract(bioclim_data_india_mcp, background.india)
    pseudo_absence_values = rbind(pseudo_absence_values.america, pseudo_absence_values.india)
    y = as.factor(c(rep(1, nrow(predictors_values_america_india)), rep(0, nrow(pseudo_absence_values))))
    sdm_data_america = cbind(y, rbind(predictors_values_america_india, pseudo_absence_values))
  }
  
  if(background_type == "sa_in"){
    background.america = randomPoints(bioclim_data_america, nrow(predictors_values_america))
    background.india = randomPoints(bioclim_data_india,nrow(predictors_values_india))
    pseudo_absence_values.america = extract(bioclim_data_america, background.america)
    pseudo_absence_values.india = extract(bioclim_data_india, background.india)
    pseudo_absence_values = rbind(pseudo_absence_values.america, pseudo_absence_values.india)
    y = as.factor(c(rep(1, nrow(predictors_values_america_india)), rep(0, nrow(pseudo_absence_values))))
    sdm_data_america = cbind(y, rbind(predictors_values_america_india, pseudo_absence_values))
  }
  
  for(j in 1:run)
  {
    train = sample(1:nrow(sdm_data_america), size = floor(0.75*nrow(sdm_data_america)), replace = FALSE)
    train_data = sdm_data_america[train,]
    test_data = sdm_data_america[-train,]
    test_y = test_data$y
    glm.fit = glm(y ~ ., data = train_data, family = binomial)
    glm.probs = predict(glm.fit, test_data, type = "response")
    glm.pred = rep("0", nrow(test_data))
    glm.pred[glm.probs > 0.5] = "1"
    m = as.matrix(table(glm.pred, test_y))
    test.error[j, i] = mean(glm.pred != test_y)
    s[j,i] = m[2,2]/(m[1,2]+m[2,2]); p = m[2,2]/(m[2,1]+m[2,2])
    f.measure[j,i] = 5*p*s[j,i]/(4*p+s[j,i])
    selected_logistic = summary(glm.fit)$coefficients[,4][-1]<0.05
    val_logistic = selected_logistic + val_logistic
    auc_logistic[j,i] = roc(test_y, glm.probs)$auc
  }
}

if(background_type == "sa"){
  write.table(f.measure, "output/fmeasure_logistic_sa_full.txt", append = FALSE)
  write.table(s, "output/sensitivity_logistic_sa_full.txt", append = FALSE)
  write.table(val_logistic, "output/values_logistic_sa_full.txt", append = FALSE)
  write.table(auc_logistic, "output/auc_logictic_sa_full.txt", append = FALSE)
  
}

if(background_type == "sa_in"){
  write.table(f.measure, "output/fmeasure_logistic_sa_in_full.txt", append = FALSE)
  write.table(s, "output/sensitivity_logistic_sa_in_full.txt", append = FALSE)
  write.table(val_logistic, "output/values_logistic_sa_in_full.txt", append = FALSE)
  write.table(auc_logistic, "output/auc_logictic_sa_in_full.txt", append = FALSE)
}

if(background_type == "sa_mcp"){
  write.table(f.measure, "output/fmeasure_logistic_sa_mcp.txt", append = FALSE)
  write.table(s, "output/sensitivity_logistic_sa_mcp.txt", append = FALSE)
  write.table(val_logistic, "output/values_logistic_sa_mcp.txt", append = FALSE)
  write.table(auc_logistic, "output/auc_logictic_sa_mcp.txt", append = FALSE)
}

if(background_type == "sa_in_mcp"){
  write.table(f.measure, "output/fmeasure_logistic_sa_in_mcp.txt", append = FALSE)
  write.table(s, "output/sensitivity_logistic_sa_in_mcp.txt", append = FALSE)
  write.table(val_logistic, "output/values_logistic_sa_in_mcp.txt", append = FALSE)
  write.table(auc_logistic, "output/auc_logictic_sa_in_mcp.txt", append = FALSE)
}
##############################
#Run of ridge and lasso regression
##############################
alpha = 0 # for ridge regression
#alpha = 1 # for lasso regression
for(i in 1:n)
{
  if(background_type == "sa"){
    background = randomPoints(bioclim_data_america, nrow(predictors_values_america))
    pseudo_absence_values = extract(bioclim_data_america, background)
    y = as.factor(c(rep(1, nrow(predictors_values_america)), rep(0, nrow(pseudo_absence_values))))
    sdm_data_america = cbind(y, rbind(predictors_values_america, pseudo_absence_values))
  }
  
  if(background_type == "sa_mcp"){
    background = randomPoints(bioclim_data_mcp, nrow(predictors_values_america))
    pseudo_absence_values = extract(bioclim_data_mcp, background)
    y = as.factor(c(rep(1, nrow(predictors_values_america)), rep(0, nrow(pseudo_absence_values))))
    sdm_data_america = cbind(y, rbind(predictors_values_america, pseudo_absence_values))
  }
  
  if(background_type == "sa_in_mcp"){
    background.america = randomPoints(bioclim_data_mcp, nrow(predictors_values_america))
    background.india = randomPoints(bioclim_data_india_mcp,nrow(predictors_values_india))
    pseudo_absence_values.america = extract(bioclim_data_mcp, background.america)
    pseudo_absence_values.india = extract(bioclim_data_india_mcp, background.india)
    pseudo_absence_values = rbind(pseudo_absence_values.america, pseudo_absence_values.india)
    y = as.factor(c(rep(1, nrow(predictors_values_america_india)), rep(0, nrow(pseudo_absence_values))))
    sdm_data_america = cbind(y, rbind(predictors_values_america_india, pseudo_absence_values))
  }
  
  if(background_type == "sa_in"){
    background.america = randomPoints(bioclim_data_america, nrow(predictors_values_america))
    background.india = randomPoints(bioclim_data_india,nrow(predictors_values_india))
    pseudo_absence_values.america = extract(bioclim_data_america, background.america)
    pseudo_absence_values.india = extract(bioclim_data_india, background.india)
    pseudo_absence_values = rbind(pseudo_absence_values.america, pseudo_absence_values.india)
    y = as.factor(c(rep(1, nrow(predictors_values_america_india)), rep(0, nrow(pseudo_absence_values))))
    sdm_data_america = cbind(y, rbind(predictors_values_america_india, pseudo_absence_values))
  }
  
  x = model.matrix(y~., data = sdm_data_america)[,-1]
  y = sdm_data_america$y
  
  grid = 10^seq(10, -2, length = 100)
  for(j in 1:run)
  {
    train = sample(1:nrow(x), size = floor(0.75*nrow(x)), replace = FALSE)
    test = (-train)
    test_y = y[test]
    
    cv.out = cv.glmnet(x = x[train,], y = as.factor(y[train]), family = "binomial", alpha=alpha, type.measure = "class")
    bestlm[j,i] = cv.out$lambda.min
    ridge.probs = predict(cv.out, s=cv.out$lambda.min, newx=x[test,], family = "binomial", type = "response")
    ridge.pred = rep(0, length(y[test]))
    ridge.pred[ridge.probs > 0.5] = 1
    selected_ridge = abs(coef(cv.out)[,1])>0.001
    val_ridge = selected_ridge+val_ridge
    m = as.matrix(table(ridge.pred, y[test]))
    test.error[j, i] = mean(ridge.pred != y[test])
    s[j,i] = m[2,2]/(m[1,2]+m[2,2]); p = m[2,2]/(m[2,1]+m[2,2])
    f.measure[j,i] = 5*p*s[j,i]/(4*p+s[j,i])
    auc_ridge[j,i] = roc(test_y, ridge.probs)$auc
    
  }
}

if(alpha == 0)
{
  if(background_type == "sa"){
    write.table(f.measure, "output/fmeasure_ridge_sa_full.txt", append = FALSE)
    write.table(s, "output/sensitivity_ridge_sa_full.txt", append = FALSE)
    write.table(val_ridge, "output/values_ridge_sa_full.txt", append = FALSE)
    write.table(auc_ridge, "output/auc_ridge_sa_full.txt", append = FALSE)
  }
  
  if(background_type == "sa_in"){
    write.table(f.measure, "output/fmeasure_ridge_sa_in_full.txt", append = FALSE)
    write.table(s, "output/sensitivity_ridge_sa_in_full.txt", append = FALSE)
    write.table(val_ridge, "output/values_ridge_sa_in_full.txt", append = FALSE)
    write.table(auc_ridge, "output/auc_ridge_sa_in_full.txt", append = FALSE)
  }
  
  if(background_type == "sa_mcp"){
    write.table(f.measure, "output/fmeasure_ridge_sa_mcp.txt", append = FALSE)
    write.table(s, "output/sensitivity_ridge_sa_mcp.txt", append = FALSE)
    write.table(val_ridge, "output/values_ridge_sa_mcp.txt", append = FALSE)
    write.table(auc_ridge, "output/auc_ridge_sa_mcp.txt", append = FALSE)
  }
  
  if(background_type == "sa_in_mcp"){
    write.table(f.measure, "output/fmeasure_ridge_sa_in_mcp.txt", append = FALSE)
    write.table(s, "output/sensitivity_ridge_sa_in_mcp.txt", append = FALSE)
    write.table(val_ridge, "output/values_ridge_sa_in_mcp.txt", append = FALSE)
    write.table(auc_ridge, "output/auc_ridge_sa_in_mcp.txt", append = FALSE)
  }
  
}

if(alpha == 1)
{
  if(background_type == "sa"){
    write.table(f.measure, "output/fmeasure_lasso_sa_full.txt", append = FALSE)
    write.table(s, "output/sensitivity_lasso_sa_full.txt", append = FALSE)
    write.table(val_ridge, "output/values_lasso_sa_full.txt", append = FALSE)
    write.table(auc_ridge, "output/auc_lasso_sa_full.txt", append = FALSE)
  }
  
  if(background_type == "sa_in"){
    write.table(f.measure, "output/fmeasure_lasso_sa_in_full.txt", append = FALSE)
    write.table(s, "output/sensitivity_lasso_sa_in_full.txt", append = FALSE)
    write.table(val_ridge, "output/values_lasso_sa_in_full.txt", append = FALSE)
    write.table(auc_ridge, "output/auc_lasso_sa_in_full.txt", append = FALSE)
  }
  
  if(background_type == "sa_mcp"){
    write.table(f.measure, "output/fmeasure_lasso_sa_mcp.txt", append = FALSE)
    write.table(s, "output/sensitivity_lasso_sa_mcp.txt", append = FALSE)
    write.table(val_ridge, "output/values_lasso_sa_mcp.txt", append = FALSE)
    write.table(auc_ridge, "output/auc_lasso_sa_mcp.txt", append = FALSE)
  }
  
  if(background_type == "sa_in_mcp"){
    write.table(f.measure, "output/fmeasure_lasso_sa_in_mcp.txt", append = FALSE)
    write.table(s, "output/sensitivity_lasso_sa_in_mcp.txt", append = FALSE)
    write.table(val_ridge, "output/values_lasso_sa_in_mcp.txt", append = FALSE)
    write.table(auc_ridge, "output/auc_lasso_sa_in_mcp.txt", append = FALSE)
  }
}
#Now we run the above codes using the selected variables in  logistic regression model.
n = 10     # Number of background to be chosen
run = 30   # Number of time the model should be run for average test error
auc_logistic = matrix(NA, nrow = run, ncol = n)
background_type = "sa"  #sa_mcp, sa_in, sa_in_mcp
# sa = South America; 
# sa_mcp = South America minimum convex polygon
# sa_in = South America and India
# sa_in_mcp = South America mcp and India mcp 
#########################
#Run of the logistic regression  
#########################
for(i in 1:n)
{
  if(background_type == "sa"){
    background = randomPoints(bioclim_data_america, 10000)
    pseudo_absence_values = extract(bioclim_data_america, background)
    y = as.factor(c(rep(1, nrow(predictors_values_america)), rep(0, nrow(pseudo_absence_values))))
    sdm_data_america = cbind(y, rbind(predictors_values_america, pseudo_absence_values))
  }
  
  if(background_type == "sa_mcp"){
    background = randomPoints(bioclim_data_mcp, 10000)
    pseudo_absence_values = extract(bioclim_data_mcp, background)
    y = as.factor(c(rep(1, nrow(predictors_values_america)), rep(0, nrow(pseudo_absence_values))))
    sdm_data_america = cbind(y, rbind(predictors_values_america, pseudo_absence_values))
  }
  
  if(background_type == "sa_in_mcp"){
    background.america = randomPoints(bioclim_data_mcp, 10000)
    background.india = randomPoints(bioclim_data_india_mcp,10000)
    pseudo_absence_values.america = extract(bioclim_data_mcp, background.america)
    pseudo_absence_values.india = extract(bioclim_data_india_mcp, background.india)
    pseudo_absence_values = rbind(pseudo_absence_values.america, pseudo_absence_values.india)
    y = as.factor(c(rep(1, nrow(predictors_values_america_india)), rep(0, nrow(pseudo_absence_values))))
    sdm_data_america = cbind(y, rbind(predictors_values_america_india, pseudo_absence_values))
  }
  
  if(background_type == "sa_in"){
    background.america = randomPoints(bioclim_data_america, 10000)
    background.india = randomPoints(bioclim_data_india,10000)
    pseudo_absence_values.america = extract(bioclim_data_america, background.america)
    pseudo_absence_values.india = extract(bioclim_data_india, background.india)
    pseudo_absence_values = rbind(pseudo_absence_values.america, pseudo_absence_values.india)
    y = as.factor(c(rep(1, nrow(predictors_values_america_india)), rep(0, nrow(pseudo_absence_values))))
    sdm_data_america = cbind(y, rbind(predictors_values_america_india, pseudo_absence_values))
  }
  for(j in 1:run)
  {
    train = sample(1:nrow(sdm_data_america), size = floor(0.75*nrow(sdm_data_america)), replace = FALSE)
    train_data = sdm_data_america[train,]
    test_data = sdm_data_america[-train,]
    test_y = test_data$y
    glm.fit = glm(y ~ bio2+bio3+bio5+bio7+bio13+bio15+bio18, data = train_data, family = binomial)
    glm.probs = predict(glm.fit, test_data, type = "response")
    glm.pred = rep("0", nrow(test_data))
    glm.pred[glm.probs > 0.5] = "1"
    test.error[j, i] = mean(glm.pred != test_y)
    auc_logistic[j,i] = roc(test_y, glm.probs)$auc
  }
}

if(background_type == "sa"){
  write.table(auc_logistic, "regularized-output/auc_logistic_sa_full.txt", append = FALSE)
}

if(background_type == "sa_in"){
  write.table(auc_logistic, "regularized-output/auc_logistic_sa_in_full.txt", append = FALSE)
}

if(background_type == "sa_mcp"){
  write.table(auc_logistic, "regularized-output/auc_logistic_sa_mcp.txt", append = FALSE)
}

if(background_type == "sa_in_mcp"){
  write.table(auc_logistic, "regularized-output/auc_logistic_sa_in_mcp.txt", append = FALSE)
}
#The codes used for predicting on India: 
#Using the selected variables:
out = matrix(NA, nrow = run, ncol = 167245)
for(i in 1:run){
  background.america = randomPoints(bioclim_data_america, nrow(predictors_values_america))
  background.india = randomPoints(bioclim_data_india,nrow(predictors_values_india))
  pseudo_absence_values.america = extract(bioclim_data_america, background.america)
  pseudo_absence_values.india = extract(bioclim_data_india, background.india)
  pseudo_absence_values = rbind(pseudo_absence_values.america, pseudo_absence_values.india)
  y = as.factor(c(rep(1, nrow(predictors_values_america_india)), rep(0, nrow(pseudo_absence_values))))
  sdm_data_america = cbind(y, rbind(predictors_values_america_india, pseudo_absence_values))
  glm.fit = glm(y ~ bio2+bio3+bio5+bio7+bio13+bio15+bio18, data = sdm_data_america, family = binomial)
  raster_pred = predict(bioclim_data_india, glm.fit, type = "response") 
  out[i,] = rasterToPoints(raster_pred)[,3]
}
average_raster_pred = rasterize(rasterToPoints(raster_pred)[,1:2], raster_pred, colMeans(out), update = TRUE)
rcl = reclassify(average_raster_pred, c(0.004143759, 0.0865, 1, 0.08651, 0.9999979, 2))
#Using all of the variables:
out_all = matrix(NA, nrow = run, ncol = 167245)
for(i in 1:run){
  #background.america = randomPoints(bioclim_data_america,10000)
  background.america = randomPoints(bioclim_data_america, nrow(predictors_values_america))
  background.india = randomPoints(bioclim_data_india,nrow(predictors_values_india))
  #background.india = randomPoints(bioclim_data_india,10000)
  pseudo_absence_values.america = extract(bioclim_data_america, background.america)
  pseudo_absence_values.india = extract(bioclim_data_india, background.india)
  pseudo_absence_values = rbind(pseudo_absence_values.america, pseudo_absence_values.india)
  y = as.factor(c(rep(1, nrow(predictors_values_america_india)), rep(0, nrow(pseudo_absence_values))))
  sdm_data_america = cbind(y, rbind(predictors_values_america_india, pseudo_absence_values))
  glm.fit_all = glm(y ~ ., data = sdm_data_america, family = binomial)
  raster_pred_all = predict(bioclim_data_india, glm.fit_all, type = "response") 
  out_all[i,] = rasterToPoints(raster_pred_all)[,3]
}
average_raster_pred_all = rasterize(rasterToPoints(raster_pred_all)[,1:2], raster_pred_all, colMeans(out_all), update = TRUE)
rcl_all = reclassify(average_raster_pred_all, c(0.004143759, 0.0865, 1, 0.08651, 0.9999979, 2))

#The following codes are used to generate figures:
#Figure 1
#This plot highlights the native range and alien range of the species on world map.
require(rworldmap)
nativeCountries = c("BLZ", "ARG", "CRI" )
alienCountries = c("IND", "PHL", "CHN")
statusDF = data.frame(country = c(nativeCountries, alienCountries),  status = c(rep("native", length(nativeCountries)), rep("alien", length(alienCountries))))
statusMap = joinCountryData2Map(statusDF, joinCode = "ISO3",
                                nameJoinColumn = "country")
mapCountryData(statusMap, nameColumnToPlot="status", catMethod = "categorical",  missingCountryCol = gray(.8), mapTitle = "Native and alien range of M. micrantha")
box()

#Figure 2
#Plots of the four different choices of background types :
  plot(india, lwd =1)
points(mic_gbif_india, col= "red", cex=0.9, pch=20)
points(background_india, pch=20, col='gray', cex=0.9)

plot(Min_convex_poly_extended_india, lwd =1)
points(mic_gbif_india, col= "red", cex=0.9, pch=20)
points(background_india, pch=20, col='gray', cex=0.9)

plot(south_america, lwd =1)
points(mic_gbif_america, col= "red", cex=0.9, pch=20)
points(background, pch=".", col='gray')

plot(Min_convex_poly_extended, lwd =1)
points(mic_gbif_america, col= "red", cex=0.9, pch=20)
points(background, pch=".", col='gray')

#Figure 3
plot(average_raster_pred)# The predicted probabilities of presence in India modeled using all variables.
hist(colMeans(out), probability=TRUE, col="grey", xlab = "Predicted probabilities", main = "")# Histogram of predicted probabilities using selected variables as predictors

plot(average_raster_pred_all) # The predicted probabilities of presence in India modeled using selected variables.
hist(colMeans(out_all), probability=TRUE, col="grey", xlab = "Predicted probabilities", main = "") # Histogram of predicted probabilities using all variables as predictors

#Figure  4
plot(rcl) # Raster plot predicted using only selected variables in the logistic regression model.
plot(rcl_all) # Raster plot predicted using all variables in the logistic regression model.


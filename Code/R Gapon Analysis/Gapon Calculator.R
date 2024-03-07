#Gapon Coefficient Calculator for ECe v. ssat analysis
#Created by A.J. Brown (ansleybrown1337@gmail.com)
#on 9/11/18

CPUname <- "AJ-CPU" #either ansleybr or AJ-CPU
Directory <- paste("C:/Users/", CPUname,"/OneDrive/Ph. D. Dissertation/Fairmont District Data/Soils Data/Soil Fertility Data/ECe v. ssat/R Gapon Analysis", sep="")

nutrients= read.csv(file=paste(Directory,"/R input for Gapon.csv", sep=""))
#nutrients = read.csv(file.choose())
View(nutrients)
attach(nutrients)

GaponFunction = function(){
  colnames(nutrients)[colnames(nutrients)=="FID"] <- "Field"
  nutrients$soil_type = seq.int(nrow(nutrients))
  #Gravimetric to Volumetric Water Content Conversion
  nutrients$ssatVWC = SoilGWC*BulkD
  nutrients$PasteVWC = PasteGWC*BulkD

  #Absorbed Concentrations in meq/kg and in eq/kg
  nutrients$soilKmeqkg=CEC*(cecK/100)*10
  nutrients$soilCameqkg=CEC*(cecCa/100)*10
  nutrients$soilMgmeqkg=CEC*(cecMg/100)*10
  nutrients$soilNameqkg=CEC*(cecNa/100)*10

  nutrients$soilKeqkg= nutrients$soilKmeqkg/1000
  nutrients$soilCaeqkg= nutrients$soilCameqkg/1000
  nutrients$soilMgeqkg= nutrients$soilMgmeqkg/1000
  nutrients$soilNaeqkg= nutrients$soilNameqkg/1000

  #Soil ESP and ESR
  nutrients$soilESP = nutrients$soilNameqkg/CEC*100
  nutrients$soilESR = nutrients$soilNameqkg/(CEC-nutrients$soilNameqkg)

  #ssat Solution concentrations in meq/L
  nutrients$ssatKmeqL = ssatK*.0256
  nutrients$ssatCameqL = ssatCa*.0499
  nutrients$ssatMgmeqL = ssatMg*.0823
  nutrients$ssatNameqL = ssatNa*.04348
  nutrients$ssatClmeqL = ssatCl*.02817
  nutrients$ssatSmeqL = ssatS*.02083
  nutrients$ssatNO3meqL = ssatNO3*.0161
  nutrients$ssatAlkmeqL = ssatAlk*.01639 #assumed to be HCO3, caCO3 = 0.01998

  #ssat Nutrients in mole/L
  nutrients$ssatKmolL = nutrients$ssatKmeqL/1000
  nutrients$ssatCamolL = nutrients$ssatCameqL/2000
  nutrients$ssatMgmolL = nutrients$ssatMgmeqL/2000
  nutrients$ssatNamolL = nutrients$ssatNameqL/1000
  nutrients$ssatClmolL = nutrients$ssatClmeqL/1000
  nutrients$ssatSmolL = nutrients$ssatSmeqL/2000
  nutrients$ssatNO3molL = nutrients$ssatNO3meqL/1000
  nutrients$ssatAlkmolL = nutrients$ssatAlkmeqL/1000

  #ssat Estimated Ionic Strength, EC, SAR, ESP, ESR, and Kgap (mol/L)^-0.5
  nutrients$ssatIonicStrength = (nutrients$ssatKmolL+
                                    nutrients$ssatCamolL*4+
                                    nutrients$ssatMgmolL*4+
                                    nutrients$ssatNamolL+
                                    nutrients$ssatClmolL+
                                    nutrients$ssatSmolL*4+
                                    nutrients$ssatNO3molL+
                                    nutrients$ssatAlkmolL)*0.5
  nutrients$ssatPredictedEC = nutrients$ssatIonicStrength/0.0127
  nutrients$ssatSAR = (nutrients$ssatNameqL/((nutrients$ssatCameqL+nutrients$ssatMgmeqL)/2)^0.5)

  #ssat Debye-Huckel Activity Coefficients
  nutrients$ssatDBCa = 10^(-0.5*4*(nutrients$ssatIonicStrength^0.5)/
                             (1+0.33*0.6*(nutrients$ssatIonicStrength^0.5)))
  nutrients$ssatDBMg = 10^(-0.5*4*(nutrients$ssatIonicStrength^0.5)/
                             (1+0.33*0.8*(nutrients$ssatIonicStrength^0.5)))
  nutrients$ssatDBNa = 10^(-0.5*1*(nutrients$ssatIonicStrength^0.5)/
                             (1+0.33*0.45*(nutrients$ssatIonicStrength^0.5)))
  nutrients$ssatDBK = 10^(-0.5*1*(nutrients$ssatIonicStrength^0.5)/
                            (1+0.33*0.3*(nutrients$ssatIonicStrength^0.5)))

  #ssat Gapon Coefficients
  nutrients$ssatGaponCaMg = nutrients$soilMgeqkg*((nutrients$ssatDBCa*nutrients$ssatCamolL)^0.5)/
    (nutrients$soilCaeqkg)/((nutrients$ssatDBMg*nutrients$ssatMgmolL)^0.5)
  nutrients$ssatGaponCaNa = nutrients$soilCaeqkg*(nutrients$ssatDBNa*nutrients$ssatNamolL)/
    (nutrients$soilNaeqkg)/((nutrients$ssatDBCa*nutrients$ssatCamolL)^0.5)
  nutrients$ssatGaponCaK = nutrients$soilCaeqkg*(nutrients$ssatDBK*nutrients$ssatKmolL)/
    (nutrients$soilKeqkg)/((nutrients$ssatDBCa*nutrients$ssatCamolL)^0.5)

  #Saturated Paste Solution Concentrations in meq/L
  #https://ascelibrary.org/doi/pdf/10.1061/9780784411698.apa
  nutrients$PasteKmeqL = PasteK*.0256
  nutrients$PasteCameqL = PasteCa*.0499
  nutrients$PasteMgmeqL = PasteMg*.0823
  nutrients$PasteNameqL = PasteNa*.04348
  nutrients$PasteClmeqL = PasteCl*.02817
  nutrients$PasteSmeqL = PasteS*.02083
  nutrients$PasteNO3meqL = PasteNO3*.0161
  nutrients$PasteAlkmeqL = PasteAlk*.01639 #assumed to be HCO3 #CaCO3 = 0.01998

  #Saturated Paste Nutrients in mole/L
  nutrients$PasteKmolL = nutrients$PasteKmeqL/1000
  nutrients$PasteCamolL = nutrients$PasteCameqL/2000
  nutrients$PasteMgmolL = nutrients$PasteMgmeqL/2000
  nutrients$PasteNamolL = nutrients$PasteNameqL/1000
  nutrients$PasteClmolL = nutrients$PasteClmeqL/1000
  nutrients$PasteSmolL = nutrients$PasteSmeqL/2000
  nutrients$PasteNO3molL = nutrients$PasteNO3meqL/1000
  nutrients$PasteAlkmolL = nutrients$PasteAlkmeqL/1000

  #Saturated Paste Estimated Ionic Strength and EC
  nutrients$PasteIonicStrength = (nutrients$PasteKmolL+
                                   nutrients$PasteCamolL*4+
                                   nutrients$PasteMgmolL*4+
                                   nutrients$PasteNamolL+
                                   nutrients$PasteClmolL+
                                   nutrients$PasteSmolL*4+
                                   nutrients$PasteNO3molL+
                                   nutrients$PasteAlkmolL)*0.5
  nutrients$PastePredictedEC = nutrients$PasteIonicStrength/0.0127
  nutrients$PasteSAR = (nutrients$PasteNameqL/((nutrients$PasteCameqL+nutrients$PasteMgmeqL)/2)^0.5)

  #Saturated Paste Debye-Huckel Activity Coefficients
  nutrients$PasteDBCa = 10^(-0.5*4*(nutrients$PasteIonicStrength^0.5)/
                              (1+0.33*0.6*(nutrients$PasteIonicStrength^0.5)))
  nutrients$PasteDBMg = 10^(-0.5*4*(nutrients$PasteIonicStrength^0.5)/
                              (1+0.33*0.8*(nutrients$PasteIonicStrength^0.5)))
  nutrients$PasteDBNa = 10^(-0.5*1*(nutrients$PasteIonicStrength^0.5)/
                              (1+0.33*0.45*(nutrients$PasteIonicStrength^0.5)))
  nutrients$PasteDBK = 10^(-0.5*1*(nutrients$PasteIonicStrength^0.5)/
                             (1+0.33*0.3*(nutrients$PasteIonicStrength^0.5)))

  #Saturated Paste Gapon Coefficients
  nutrients$PasteGaponCaMg = nutrients$soilMgeqkg*((nutrients$PasteDBCa*nutrients$PasteCamolL)^0.5)/
   (nutrients$soilCaeqkg)/((nutrients$PasteDBMg*nutrients$PasteMgmolL)^0.5)
  nutrients$PasteGaponCaNa = nutrients$soilCaeqkg*(nutrients$PasteDBNa*nutrients$PasteNamolL)/
   (nutrients$soilNaeqkg)/((nutrients$PasteDBCa*nutrients$PasteCamolL)^0.5)
  nutrients$PasteGaponCaK = nutrients$soilCaeqkg*(nutrients$PasteDBK*nutrients$PasteKmolL)/
   (nutrients$soilKeqkg)/((nutrients$PasteDBCa*nutrients$PasteCamolL)^0.5)

  ## Create output file name
  Exportname <- paste(Directory,"/", paste("Gapon Output", Sys.Date(),".csv", sep=""), sep = "")
  ## Check name for correctness
  print(Exportname)
  ## Export .csv master file
  write.csv(nutrients, file = Exportname, row.names = F)
}
GaponFunction()

#important correlations
nut2 = nutrients[, c(5:ncol(nutrients))]
cor(nut2)
x1 = nutrients$PasteNa
x2 = nutrients$ssatNa
graph_output = function(){
  plot(x1, nutrients$ECe,
       xlim = c(0,10000),
       ylim = c(0,30),
       xlab = 'Solution Sulfur (ppm)',
       ylab = 'Electrical Conductivity (dS/m)')
  points(x2, nutrients$ECpw, pch = 16)
  model1 = lm(nutrients$ECe~x1)
  abline(model1$coefficients, lty=2)
  model2 = lm(nutrients$ECpw~x2)
  abline(model2$coefficients, lty=1)
  legend('topleft', legend = c('Saturated Paste Extract', 'SSAT Pore Water'),
         lty = c(2,1), pch = c(1,16), title = 'Solution Types')
  print(summary(model1))
  summary(model2)
}
graph_output()

graph_output2 = function(){
  plot(x1, nutrients$ECe/nutrients$ECpw,
       xlim = c(0,2000),
       #ylim = c(0,12),
       xlab = 'Solution Sulfur (ppm)',
       ylab = 'Electrical Conductivity (dS/m)')
  points(x2, nutrients$ECe/nutrients$ECpw, pch = 16)
  model1 = lm(nutrients$ECe/nutrients$ECpw~x1)
  abline(model1$coefficients, lty=2)
  model2 = lm(nutrients$ECe/nutrients$ECpw~x2)
  abline(model2$coefficients, lty=1)
  legend('bottomright', legend = c('Saturated Paste Extract', 'SSAT Pore Water'),
         lty = c(2,1), pch = c(1,16), title = 'Solution Types')
  print(summary(model1))
  summary(model2)
}
graph_output2()

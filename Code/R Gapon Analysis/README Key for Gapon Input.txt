This document serves as a key for identifying variables in the "R input for Gapon" and "Gapon Output" files.

Key:
INPUT$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
BulkD - bulk density (g/cm3)
ECe - saturated paste EC (dS/m)
ECpw - pore water EC from SSAT (dS/m)
SoilGWC - Field soil gravimetric water content (%)
PasteGWC - saturated paste gravimetric water content (%)
SSATNa - SSAT sodium (ppm)
SSATCa - SSAT calicium (ppm)
SSATMg - SSAT magnesium (ppm)
SSATK - SSAT potassium (ppm)
SSATAlk - SSAT Hardness/Alkalinity (ppm)
SSATNO3 - SSAT nitrate (ppm)
SSATS - SSAT sulfer (ppm)
SSATCl - SSAT chloride (ppm)
PasteNa - Paste sodium (ppm) 
PasteCa - Paste calicium (ppm)
PasteMg - Paste magnesium (ppm)
PasteK - Paste potassium (ppm)
PasteAlk - Paste Hardness/Alkalinity (ppm)
PasteNO3 - Paste nitrate (ppm)
PasteS - Paste sulfer (ppm)
PasteCl - Paste chloride (ppm)
SoilCa - Soil calcium (ppm)
SoilMg - Soil magnesium (ppm)
SoilNa - Soil sodium (ppm)
SoilK - Soil potassium (ppm)
SoilNO3 - Soil nitrate (ppm)
SoilS - Soil sulfur (ppm)
CEC - Sum of cations (me/100g)
cecH - hydrogren saturation of cations (%)
cecK - potassium saturation of cations (%)
cecCa - calcium saturation of cations (%)
cecMg - magnesium saturation of cations (%)
cecNa - sodium saturation of cations (%)
Qs - saturation volumetric water content for Van Genuchten curve
Qr - residual volumetric water content for Van Genuchten curve
alpha - shape parameter for Van Genuchten curve
n - shape parameter for Van Genuchten curve
m - shape parameter for Van Genuchten curve

OUTPUT$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
ssatVWC - SSAT volumetric water content (m3/m3)
soilKmeqkg - soilpotassium (meq/kg)
soilCameqkg - soilcalcium (meq/kg)
soilMgmeqkg - soilmagnesium (meq/kg)
soilNameqkg - soilsodium (meq/kg)
soilKeqkg - soilpotassium (eq/kg)
soilCaeqkg - soilcalcium (eq/kg)
soilMgeqkg - soilmagnesium (eq/kg)
soilNaeqkg - soilsodium (eq/kg)
ssatKmeqL - SSATpotassium (meq/L)
ssatCameqL - SSATcalcium (meq/L)
ssatMgmeqL - SSATmagnesium (meq/L)
ssatNameqL - SSATsodium (meq/L)
ssatClmeqL - SSATchloride (meq/L)
ssatSmeqL - SSATsulfur (meq/L)
ssatNO3meqL - SSATnitrate (meq/L)
PasteKmeqL - Pastepotassium (meq/L)
PasteCameqL - Pastecalcium (meq/L)
PasteMgmeqL - Pastemagnesium (meq/L)
PasteNameqL - Pastesodium (meq/L)
PasteClmeqL - Pastechloride (meq/L)
PasteSmeqL - Pastesulfur (meq/L)
PasteNO3meqL - Pastenitrate (meq/L)
ssatKmolL - SSATpotassium (mol/L)
ssatCamolL - SSATcalcium (mol/L)
ssatMgmolL - SSATmagnesium (mol/L)
ssatNamolL - SSATsodium (mol/L)
ssatClmolL - SSATchloride (mol/L)
ssatSmolL - SSATsulfur (mol/L)
ssatNO3molL - SSATnitrate (mol/L)
PasteKmolL - Pastepotassium (mol/L)
PasteCamolL - Pastecalcium (mol/L)
PasteMgmolL - Pastemagnesium (mol/L)
PasteNamolL - Pastesodium (mol/L)
PasteClmolL - Pastechloride (mol/L)
PasteSmolL - Pastesulfur (mol/L)
PasteNO3molL - Pastenitrate (mol/L)
ssatIonicStrength - SSAT- (mol/L)
ssatPredictedEC - SSAT- (dS/m)
ssatSAR - SSAT- ((meq/L)^.5)
PasteIonicStrength - Paste- (mol/L)
PastePredictedEC - Paste- (dS/m)
soilESP - Soil- (%)
soilESR - Soil- (unitless)
ssatDBCa - SSATcalcium (Debye-Huckel coeff)
ssatDBMg - SSATmagnesium (Debye-Huckel coeff)
ssatDBNa - SSATsodium (Debye-Huckel coeff)
ssatDBK - SSATpotassium (Debye-Huckel coeff)
PasteDBCa - Pastecalcium (Debye-Huckel coeff)
PasteDBMg - Pastemagnesium (Debye-Huckel coeff)
PasteDBNa - Pastesodium (Debye-Huckel coeff)
PasteDBK - Pastepotassium (Debye-Huckel coeff)
ssatGaponCaMg - SSATCa/Mg (Gapon Selectivity Coefficient)
ssatGaponCaNa - SSATCa/Na (Gapon Selectivity Coefficient)
ssatGaponCaK - SSATCa/K (Gapon Selectivity Coefficient)
PasteGaponCaMg - PasteCa/Mg (Gapon Selectivity Coefficient)
PasteGaponCaNa - PasteCa/Na (Gapon Selectivity Coefficient)
PasteGaponCaK - PasteCa/K (Gapon Selectivity Coefficient)

Additional Notes about Hydrus Modelling:
-Should we change precipitated amounts of gypsum/CaCO3 for looking at EC changes? Or maybe Adsorbed? Does it matter?
-Should we change the Gapon coefficients when switching from paste to SSAT?
***Ans
-Should we assume a bulk density for the sat paste to calculate VWC?

-How do I get Alkalinity into meq/L from ppm of hardness?
***Answer: Hydrus manual says sum of HCO3 and CO3 (so for us, just HCO3 was reported)
-How to we know precipitated amounts of Gypsum? CaCO3? 
***Answer: We don't, but this is the only thing we can change in hydrus that directly relates to gypsum
-Under "Chemical Parameters" what is "Conductivity Reduction due to Chemistry?" 
***Answer: is is about hydraulic conductivity, not EC
-Under "Solute Transport Boundary Conditions" should we have the lower boundary as concentration? flux? or zero gradient? 
***Answer: Shouldn't Matter
-Is EC sensitive to initial water content VWC?
***Very small influence (DL3 120cm EC: at 0.1VWC=5.24dS/m, at 0.7VWC=5.21dS/m. WITH 20 PRECIP GYPSUM)
***NO influence         (DL3 120cm EC: at 0.1VWC=4.08dS/m, at 0.7VWC=4.08dS/m. WITH 0 PRECIP GYPSUM)
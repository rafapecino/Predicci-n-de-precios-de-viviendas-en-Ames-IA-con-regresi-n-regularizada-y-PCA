# ==============================================================================
# EJERCICIO FEEDBACK 1: PREDICCIÓN DE PRECIOS VIVIENDA (AMPLIADO)
# ==============================================================================

# 1. CARGA DE LIBRERÍAS
if(!require(pacman)) install.packages("pacman")
pacman::p_load(tidyverse, caret, glmnet, corrplot, naniar, gridExtra, moments, e1071)

# 2. CARGA DE DATOS
df <- read.csv("train.csv", stringsAsFactors = FALSE)
# Eliminamos ID y outliers extremos visuales conocidos en este dataset (GrLivArea > 4000)
df <- df %>% filter(GrLivArea < 4000) %>% select(-Id)

# ==============================================================================
# 3. ANÁLISIS EXPLORATORIO (EDA) - GENERACIÓN DE IMÁGENES PARA EL INFORME
# ==============================================================================

# A. Variable Objetivo
png("img_01_distribucion_precio.png", width=800, height=400)
par(mfrow=c(1,2))
hist(df$SalePrice, main="SalePrice Original", col="skyblue", border="white")
# Log-transformación para corregir asimetría (Skewness)
df$SalePrice <- log(df$SalePrice)
hist(df$SalePrice, main="Log(SalePrice) - Normalizada", col="lightgreen", border="white")
dev.off()

# B. Correlaciones (Solo numéricas)
nums <- unlist(lapply(df, is.numeric))
cor_matrix <- cor(df[,nums], use="complete.obs")
png("img_02_heatmap.png", width=800, height=800)
corrplot(cor_matrix, method="color", type="upper", tl.col="black", tl.cex=0.6)
dev.off()

# C. Calidad vs Precio (Boxplot)
png("img_03_calidad_precio.png", width=600, height=400)
ggplot(df, aes(x=factor(OverallQual), y=SalePrice)) + 
  geom_boxplot(fill="orange", alpha=0.7) +
  labs(title="Relación Calidad General vs Precio (Log)", x="Overall Quality", y="Log(SalePrice)") +
  theme_minimal()
dev.off()

# ==============================================================================
# 4. PREPROCESAMIENTO AVANZADO
# ==============================================================================

# --- Imputación de Nulos ---
# Categóricas donde NA = "No tiene"
cols_none <- c("Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1", 
               "BsmtFinType2","FireplaceQu","GarageType","GarageFinish", 
               "GarageQual","GarageCond","PoolQC","Fence","MiscFeature")
df[cols_none] <- lapply(df[cols_none], function(x) replace_na(x, "None"))

# LotFrontage: Imputar con la mediana por vecindario (más preciso)
df <- df %>% group_by(Neighborhood) %>% 
  mutate(LotFrontage = ifelse(is.na(LotFrontage), median(LotFrontage, na.rm=TRUE), LotFrontage)) %>%
  ungroup()

# Resto de NAs numéricos a 0 y categóricos a la moda
df[is.na(df)] <- 0 # Simplificación para ejercicio
df[sapply(df, is.character)] <- lapply(df[sapply(df, is.character)], as.factor)

# --- Skewness: Logaritmo a variables predictoras numéricas asimétricas ---
# Esto es lo que comentaba tu compañero: "logaritmo de variables no simétricas"
numeric_feats <- df[, sapply(df, is.numeric)]
skewed_feats <- sapply(numeric_feats, skewness)
skewed_feats <- skewed_feats[skewed_feats > 0.75] # Umbral de asimetría
for(x in names(skewed_feats)) {
  df[[x]] <- log1p(df[[x]]) # log(x + 1) para evitar log(0)
}

# 5. SPLIT DATA
set.seed(123)
trainIndex <- createDataPartition(df$SalePrice, p = .7, list = FALSE)
train <- df[trainIndex,]
test  <- df[-trainIndex,]

# Matrices para glmnet (crea dummies automáticos)
x_train <- model.matrix(SalePrice ~ ., train)[,-1]
y_train <- train$SalePrice
x_test  <- model.matrix(SalePrice ~ ., test)[,-1]
y_test  <- test$SalePrice

# ==============================================================================
# 6. MODELADO Y PCA (Lo que menciona tu compañero del 94.75%)
# ==============================================================================

# --- Análisis de PCA para ver cuántos componentes explican la varianza ---
pca_res <- prcomp(x_train, center = TRUE, scale. = TRUE)
var_explained <- pca_res$sdev^2 / sum(pca_res$sdev^2)
cum_var <- cumsum(var_explained)

# Gráfico de Varianza Acumulada
png("img_04_pca_variance.png", width=600, height=400)
plot(cum_var, xlab = "Nº Componentes Principales", 
     ylab = "Varianza Acumulada Explicada", type = "b", col="blue")
abline(h=0.95, col="red", lty=2) # Línea del 95%
abline(v=which(cum_var >= 0.95)[1], col="red", lty=2)
text(60, 0.8, paste("95% varianza con", which(cum_var >= 0.95)[1], "componentes"))
dev.off()

# Imprimir el resultado para el informe
cat("Número de componentes para 95% varianza:", which(cum_var >= 0.95)[1], "\n")

# --- MODELOS (Ridge, Lasso) ---
ctrl <- trainControl(method = "cv", number = 10)

# Ridge
cv_ridge <- cv.glmnet(x_train, y_train, alpha = 0)
m_ridge  <- glmnet(x_train, y_train, alpha = 0, lambda = cv_ridge$lambda.min)

# Lasso
cv_lasso <- cv.glmnet(x_train, y_train, alpha = 1)
m_lasso  <- glmnet(x_train, y_train, alpha = 1, lambda = cv_lasso$lambda.min)

# ==============================================================================
# 7. EVALUACIÓN
# ==============================================================================
eval_calc <- function(model, x, y_true){
  pred_log <- predict(model, newx = x, s = "lambda.min")
  pred_orig <- exp(pred_log)
  true_orig <- exp(y_true)
  
  rmse <- sqrt(mean((pred_orig - true_orig)^2))
  mae  <- mean(abs(pred_orig - true_orig))
  r2   <- cor(pred_orig, true_orig)^2
  return(c(RMSE=rmse, MAE=mae, R2=r2)) # Devuelve vector con métricas
}

res_ridge <- eval_calc(cv_ridge, x_test, y_test)
res_lasso <- eval_calc(cv_lasso, x_test, y_test)

# Guardar resultados en CSV para poner en la tabla de Latex
resultados <- rbind(Ridge = res_ridge, Lasso = res_lasso)
write.csv(resultados, "resultados_finales.csv")

# Gráfico Predicciones (Lasso)
p_lasso <- predict(m_lasso, newx = x_test, s = cv_lasso$lambda.min)
png("img_05_pred_vs_real.png", width=600, height=600)
plot(exp(y_test), exp(p_lasso), main="Lasso: Predicción vs Realidad", 
     xlab="Real ($)", ylab="Predicho ($)", pch=20, col="darkblue")
abline(0,1, col="red", lwd=2)
dev.off()

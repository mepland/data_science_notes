install.packages(c("survival", "survminer"))
library("survival")
library("survminer")

df <- stanford2
df$agecat <- cut(df$age, breaks=c(0,40, Inf), labels=c('Under 40', 'Over 40'), right=FALSE)
df <- df[with(df, order(time)),]
df[1:5,c('time', 'status', 'age', 'agecat', 't5')]
summary(df[c('time', 'age', 'agecat', 't5')])

km.model <- survfit(Surv(time, status) ~ agecat, data=df, type='kaplan-meier')
summary(km.model)$table

survdiff(Surv(time, status) ~ agecat, data=df)

pdf('~/stanford_km.pdf')
ggsurvplot(km.model, xlab='Time', ylab='S(t)', size = 1, linetype = 'strata', palette=c('#4e79a7', '#f28e2b'), conf.int = TRUE, legend = c(0.85, 0.85), legend.y = 1, legend.title = '', legend.labs = c('Under 40', 'Over 40'))
dev.off()

pdf('~/stanford_km_annotated.pdf')
ggsurvplot(km.model, xlab='Time', ylab='S(t)', size = 1, linetype = 'strata', palette=c('#4e79a7', '#f28e2b'), conf.int = TRUE, legend = c(0.85, 0.85), legend.y = 1, legend.title = '', legend.labs = c('Under 40', 'Over 40'),
pval = TRUE, # Add survdiff p-value
risk.table = TRUE, # Absolute number at risk
risk.table.y.text.col = FALSE, risk.table.col = "strata",
ncensor.plot = TRUE, # plot censored patients versus time
surv.median.line = "h", # add horizontal median
)
dev.off()

cox.model_age <- coxph(Surv(time, status) ~ age, data=df[!is.na(df$t5), ])
summary(cox.model_age)

pdf('~/stanford_cox_age_baseline_survival.pdf')
ggsurvplot(survfit(cox.model_age, data=df[!is.na(df$t5), ]), xlab='Time', ylab='S(t)', size = 1, linetype = 'strata', palette=c('#4e79a7'), conf.int = TRUE, legend = c(0.85, 0.85), legend.y = 1, legend.title = '', legend.labs = c('Baseline Survival'))
dev.off()

cox.model_age_t5 <- coxph(Surv(time, status) ~ age + t5, data=df[!is.na(df$t5), ])
summary(cox.model_age_t5)

anova(cox.model_age, cox.model_age_t5, test='LRT')

cox.model_age.ph <- cox.zph(cox.model_age)
cox.model_age.ph

cox.model_age_t5.ph <- cox.zph(cox.model_age_t5)
cox.model_age_t5.ph

pdf('~/stanford_cox_age_schoenfeld_residuals.pdf')
ggcoxzph(cox.model_age.ph)
dev.off()

pdf('~/stanford_cloglog.pdf')
plot(km.model, fun='cloglog', xlab='log(t)', ylab='log(-log(S(t)))', col=c('#4e79a7', '#f28e2b'))
legend('bottomright', inset=.02, legend=c('Under 40', 'Over 40'), col=c('#4e79a7', '#f28e2b'), lty=1:2, box.lty=0)
dev.off()

pdf('~/stanford_cox_age_martingale_residuals.pdf')
ggcoxdiagnostics(cox.model_age, type = "martingale", ox.scale='linear.predictions')
dev.off()

pdf('~/stanford_cox_age_martingale_residuals_age.pdf')
ggcoxfunctional(Surv(time, status) ~ age, data = df)
dev.off()

pdf('~/stanford_cox_age_deviance_residuals.pdf')
ggcoxdiagnostics(cox.model_age, type = "deviance", ox.scale='linear.predictions')
dev.off()

pdf('~/stanford_cox_age_dfbeta.pdf')
ggcoxdiagnostics(cox.model_age, type = "dfbeta", ox.scale='observation.id')
dev.off()

# export data for use in python with lifelines
write.csv(df[!is.na(df$t5), ],"~/stanford.csv", row.names = FALSE)

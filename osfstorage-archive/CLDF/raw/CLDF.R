library(tidyverse)

evid = read.csv('Evidentiality.csv')
tense = read.csv('Tense.csv')
num = read.csv('Number.csv')
form = read.csv('Form.csv')
nform = read.csv('NumberForms.csv')

# Languages

langs = bind_rows(
  evid %>% select(Language, Family, Source, Page),
  tense %>% select(Language, Family, Source) %>%
    separate(Source, into=c('Source', 'Page'), sep=':'),
  num %>% select(Language, Family, Source, Page)
)

# Evidentiality

evid = evid %>%
  mutate(Language = as.character(Language),
    Language_ID = substr(Language, nchar(Language)-2, nchar(Language)),
    Language_ID = ifelse(Language_ID %in% c('ian', 'ish', 'ano', 'ara'), 
                         substr(Language, 0, 3), 
                         Language_ID),
    Language_ID = tolower(Language_ID)) %>%
  group_by(Language_ID) %>%
  mutate(N=1:n(),
         N = as.character(N)) %>%
  unite(Language_ID, Language_ID, N, sep='')

evid_lang = evid %>%
  select(Language_ID, Name=Language, Family, Source, Page)

write.csv(evid_lang, 'Evid_languages.csv', row.names = F)


evid_features = evid %>% 
  select(Language_ID, A, H, HQ, I, IA, IAHQ, IH, Q, S, SIAHQ, V, VS, VSI) %>%
  gather(Feature, Value, A:VSI) %>%
  mutate(Value = ifelse(nchar(Value)>0, 1, 0)) %>%
  arrange(Language_ID)

write.csv(evid_features, 'Evid_features.csv', row.names = F)


# Tense

tense = tense %>%
  mutate(Language = as.character(Language),
         Language = str_replace_all(Language, "[[:punct:]]", ""),
         Language_ID = substr(Language, nchar(Language)-2, nchar(Language)),
         Language_ID = ifelse(Language_ID %in% c('ian', 'ish', 'ano', 'ara'), 
                              substr(Language, 0, 3), 
                              Language_ID),
         Language_ID = tolower(Language_ID)) %>%
  group_by(Language_ID) %>%
  mutate(N=1:n(),
         N = as.character(N)) %>%
  unite(Language_ID, Language_ID, N, sep='')

tense_lang = tense %>%
  select(Language_ID, Name=Language, Family, Source) %>%
  separate(Source, into=c('Source', 'Page'), sep=':')

write.csv(tense_lang, 'Tense_languages.csv', row.names = F)


tense_features = tense %>% 
  select(Language_ID, remote_past, recent_past, immediate_past, past_wo_remote,
         past, present, immediate_future, remote_future, future, non_past, 
         non_future, pre_hodiernal) %>%
  gather(Feature, Value, remote_past:pre_hodiernal) %>%
  mutate(Value = ifelse(nchar(Value)>0, 1, 0)) %>%
  arrange(Language_ID)

write.csv(tense_features, 'Tense_features.csv', row.names = F)

# Number

num = num %>%
  mutate(Language = as.character(Language),
         Language = str_replace_all(Language, "[[:punct:]]", ""),
      Language_ID = substr(Language, nchar(Language)-2, nchar(Language)),
       Language_ID = ifelse(Language_ID %in% c('ian', 'ish', 'ano', 'ara', 'ese'), 
                            substr(Language, 0, 3), 
                            Language_ID),
       Language_ID = tolower(Language_ID)) %>%
  group_by(Language_ID) %>%
  mutate(N=1:n(),
         N = as.character(N)) %>%
  unite(Language_ID, Language_ID, N, sep='')

num_lang = num %>%
  select(Language_ID, Name=Language, Family, Source, Page)

write.csv(num_lang, 'Num_languages.csv', row.names = F)


num_features = num %>% 
  select(Language_ID, BARE:PL8) %>%
  gather(Feature, Value, BARE:PL8) %>%
  mutate(Value = ifelse(nchar(Value)>0, 1, 0)) %>%
  arrange(Language_ID)

write.csv(num_features, 'Num_features.csv', row.names = F)

# Forms

form_e = form %>%
  mutate(Language = as.character(Language)) %>%
  filter(Domain == 'Evidentiality') %>%
  left_join(evid_lang %>% select('Language'=Name, Language_ID, Family))

form_e_lang = form_e %>%
  select(Language_ID, Language, Source, Page) %>%
  distinct() %>%
  group_by(Language_ID) %>%
  mutate(Page = paste(Page, collapse=";")) %>%
  ungroup() %>%
  distinct()

write.csv(form_e_lang, 'evid_form_languages.csv', row.names = F)

form_e_feat = form_e %>%
  mutate(Meaning = toupper(Meaning)) %>%
  select(Language_ID, Feature=Meaning, Vale=Form)

write.csv(form_e_feat, 'evid_form_featuress.csv', row.names = F)


a = form %>% filter(Domain != 'Evidentiality') %>% pull(Language) %>% as.character() %>% unique
b = tense_lang %>% pull(Name) %>% unique()

form_t = form %>%
  mutate(Language = as.character(Language),
         Language = str_replace_all(Language, "[[:punct:]]", "")) %>%
  filter(Domain != 'Evidentiality') %>% 
  left_join(tense_lang %>% select(Language=Name, Language_ID, Family))

form_t_lang = form_t %>%
  select(Language_ID, Language, Source, Page) %>%
  distinct() %>%
  group_by(Language_ID) %>%
  mutate(Page = paste(Page, collapse=";")) %>%
  ungroup() %>%
  distinct()

write.csv(form_t_lang, 'tense_form_languages.csv', row.names = F)

form_t_feat = form_t %>%
  mutate(Meaning = toupper(Meaning)) %>%
  select(Language_ID, Feature=Meaning, Vale=Form)

write.csv(form_t_feat, 'tense_form_featuress.csv', row.names = F)


nform = nform %>%
  left_join(num_lang %>% select(Language_ID, Language=Name))

nform_feat = 
  nform %>%
    select(Language_ID, Feature=Meaning, Value=Form1)
  
write.csv(nform_feat, 'number_form_featuress.csv', row.names = F)

  
  

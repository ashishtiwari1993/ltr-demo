# LTR Demo for enagement data (clicks)

Tested on `Python 3.10.16`

## Push data to Elasticsearch (index=bollywood-movies)

```sh
curl -s -H "Authorization: ApiKey elasticsearch_api_key" -H "Content-Type: application/x-ndjson" -XPOST https://elastic_endpoint/_bulk --data-binary "@nd-movies.json"
```

## Install Dependencies

```sh
pip install -r requirements.txt
```

## Update config

```sh
cp config/sample.config.yml config/config.yml 
```

Change config accordingly.

```
vim config/config.yml
```

```yaml
elastic:
  cloud_id: "elastic_cloud_id"
  api_key: "elastic_api_key"
  index_name: "bollywood-movies"
  judgement_list_filename: "judgement_list.csv"
  model_id: "ltr-model-xgboost-movies"
```

## Run streamlit app

```sh
streamlit run app.py
```

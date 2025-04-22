# Praxisprojekt: Erstellung eines Systems zur automatischen Klassifikation in der NWBib

Das HBZ betreibt mit der [NWBib](https://nwbib.de/) seit 1983 die Landesbibliographie für das Bundesland Nordrhein-Westfalen. Diese Sammlung enthält zur Zeit mehr als 450.000 Nachweise zu Titeln mit entsprechendem Regionalbezug, die jeweils intellektuell erschlossen und klassifiziert wurden. In diesem Praxisprojekt wurde mithilfe von Methoden des maschinellen Lernens ein System zur (halb-)automatischen Klassifikation geschaffen, das für neue Titel automatisch geeignete Klassen innerhalb der [NWBib-Sachsystematik](https://nwbib.de/subjects) vorschlagen kann. Hierzu kommt das an der finnischen Nationalbibliothek entwickelte System [Annif](NWBib) zum Einsatz.

## HOWTO

Die folgende Anleitung besteht aus zwei Teilen: Im ersten Teil wird ein vollständiges Bootstrapping von Annif beschrieben, wobei die im Rahmen des Praxisprojekts erarbeiteten Daten verwendet werden, hiermit lassen sich die Projektergebnisse nachvollziehen. Im zweiten Teil werden die erstellten Tools erläutert, mit denen die Daten aufbereitet wurden, hiermit ist es möglich, neue Trainings- und Evaluationsdaten auf Grundlage eines aktuellen Abzugs der NWBib zu generieren. 

### Annif-Installation, Training und Verifikation

#### Installation (Schritte 1-3)

##### Variante A mit Docker (genutzt im hbz)

Verwendet wir im Folgenden: 

- Docker
...

1. Auf Basis der `docker-compose.yml` wird ein Container gebaut

`docker compose up` - ggf. als root mit sudo


2. Wechsel in den Container:

`sudo docker exec -it nwbib-annif-annif_app-1 bash`

3. Prüfen, ob Docker läuft:

`annif list-projects`


##### Variante B mit lokaler Annif-Installation

1. Im Folgenden wird als Grundlage das Aufsetzten einer lokalen Annif-Installation beschrieben. Falls eine dauerhafte Server-Lösung gewünscht ist, sollte Annif als WSGI-Service installiert werden, was zusätzliche Schritte erfordert. Eine Anleitung dazu findet sich [hier](https://github.com/NatLibFi/Annif/wiki/Running-as-a-WSGI-service), die folgende grundlegende Vorgehensweise sind aber bei beiden Wegen identisch.

Verwendet werden im Folgenden 

- Annif v1.3.1 (auch mit 1.4.0.dev  getestet)
- Python 3.8 + (Auch getestet mit 3.12)
- nltk 3.9.1

```bash
git clone https://github.com/NatLibFi/Annif.git
cd Annif
python3 -m venv annif-venv (kann anders lauten je nach vorhandener Python-Binary)
. venv/bin/activate
pip install pip setuptools --upgrade
pip install annif
python -m nltk.downloader punkt
pip install .[fasttext]
pip install .[omikuji]
pip install .[nn]
```

Wenn Annif nicht von GitHub installiert wurde, muss `pip install annif[fasttext]` etc. verwendet werden.

2. Anschließend sollten einige Dateien aus diesem Repository in die Annif-Installation kopiert werden, im Einzelnen sind dies die Projekt-Konfigurationsdatei, der Trainings- und der Testkorpus sowie das NWBib-SKOS-Vokabular. Die letzteren platzieren wir zwecks Übersicht in einem neuen Ordner `nwbib-data`:

```bash
mkdir -p Pfad/zu/Annif/nwbib-data
cp annif-projects/projects.cfg Pfad/zu/Annif/
cp annif-projects/nwbib-data/nwbib_subjects_train.tsv Pfad/zu/Annif/nwbib-data/
cp annif-projects/nwbib-data/nwbib_subjects_test.tsv Pfad/zu/Annif/nwbib-data/
cp annif-projects/nwbib-data/nwbib.ttl Pfad/zu/Annif/nwbib-data/
``` 

3. Wir wechseln zurück in das Annif-Verzeichnis und sollten zunächst überprüfen, ob Annif funktioniert und die Projekte korrekt erkennt (die virtuelle Python-Umgebung aus Schritt 1 muss weiterhin aktiviert sein). Wir sollten eine Reihe von `nwbib`-Backends angezeigt bekommen, die noch alle "untrainiert" sind:

```bash
cd Pfad/zu/Annif/
annif list-projects
```

### Training und Verifikation

Hinweis: Für die Docker Variante gehen wir davon aus, dass wir uns im COntainer befinden. Siehe Schritt 2 in der Dockervariante.

Bevor wir mit dem Trainieren beginnen, muss noch einmalig das NWBib-Vokabular geladen werden. Dies geschieht mittels

```bash
annif load-vocab nwbib-de nwbib-data/nwbib.ttl
```

4. Jetzt können wir Annif anweisen, die einzelnen Backends zu trainieren, indem wir ihm den Trainingskorpus übergeben. Wir beginnen mit dem einfachsten Backend, das die TF-IDF-Heuristik implementiert:

```bash
annif train nwbib-tfidf nwbib-data/nwbib_subjects_train.tsv
```

Das Training kann einige Zeit in Anspruch nehmen, in diesem Fall sollte es aber relativ schnell beendet sein. Nach dem Durchlauf können wir überprüfen, wie gut das Backend funktioniert, indem wir es mit den bislang zurückgehaltenen Titeln aus unserem Testkorpus evaluieren:

```bash
annif eval nwbib-tfidf nwbib-data/nwbib_subjects_test.tsv
```

Nach dem Durchlauf liefert Annif entsprechende Statistiken etwa zu Precision, Recall, F1-Wert und NDCG.

5. Nach dem gleichen Schema trainieren wir nun noch die verbleibenden Backends:

```bash
annif train nwbib-mllm nwbib-data/nwbib_subjects_train.tsv
annif train nwbib-omikuji nwbib-data/nwbib_subjects_train.tsv
annif train nwbib-fasttext nwbib-data/nwbib_subjects_train.tsv
```

6. Sobald die einzelnen Backends einsatzbereit sind, müssen im letzten Schritt noch die Ensemble-Backends trainiert werden. Die einfachen Ensembles benötigen kein Training, sondern lediglich die NN-basierten:

```bash
annif train nwbib-ensemble-nn nwbib-data/nwbib_subjects_train.tsv -j 8
annif train nwbib-triple-ensemble-nn nwbib-data/nwbib_subjects_train.tsv -j 8
```

Der Parameter `-j` gibt die Anzahl der maximalen Threads an, hier empfiehlt es sich, das Maximum nicht auszuschöpfen, da es sonst zu Speicher- und Performanceproblemen kommen kann. Auf der zum Training verwendeten Maschine standen 16 logische CPU-Kerne zur Verfügung, die Anzahl der Threads wurde auf die Hälfte beschränkt.

### Generierung neuer Korpora

Falls Annif mit einem aktuellen Snapshot der NWBib neu trainiert werden soll, können auf folgende Weise der entsprechende Trainings- und Testkorpus erstellt werden: 

1. Gesamtabzug der NWBib über die [lobid-API](https://blog.lobid.org/2019/10/08/nwbib-at-cdv.html) extrahieren und entpacken :

```bash
curl --header "Accept-Encoding: gzip" "http://lobid.org/resources/search?q=inCollection.id%3A%22http%3A%2F%2Flobid.org%2Fresources%2FHT014176012%23%21%22&format=jsonl" > annif-projects/nwbib-data/nwbib.gz
gunzip annif-projects/nwbib-data/nwbib.gz
```

2. Extraktor-Skript ausführen, um die Korpora zu erzeugen. Das Skript kann mit zusätzlichen Argumenten aufgerufen werden, `-h` zeigt eine Übersicht. Falls keine weiteren Einstellungen zur Korpusgröße angegeben werden, werden Trainings- und Testkorpus mit einer randomisierten 90:10-Partitionierung erstellt, dies ist die Datengrundlage, die auch im Fachaufsatz beschrieben wird. Zusätzlich sollte allerdings das SKOS-Vokabular der NWBib über den Parameter `-v` angeben werden, damit fehlerhafte Terme ausgefiltert werden können.

```bash
cd annif-projects/nwbib-data/
python nwbib_extractor.py -v nwbib.ttl
```

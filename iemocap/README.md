## The IEMOCAP Dataset

The Interactive Emotional Dyadic Motion Capture (IEMOCAP) database is an acted, multimodal and multispeaker database.
It consists of 12 hours of audiovisual data, including video, speech, motion capture of face, text transcriptions.

Sessions were manually segmented into utterances, spoken by 10 (5 female and 5 male) professional actors in fluent English. Each utterance was annotated by at least 3 human annotators in a few categorical attributes:
* Anger
* Happiness
* Excitement
* Sadness
* Frustration
* Fear
* Surprise
* Other
* Neutral

And in 3 dimensional attributes:
* Valence
* Activation
* Dominance

To collect the data, the sessions were organized with emotion elicitation techniques such as improvisations and scripts, which makes the dataset better for the task of analysis of human dyadic interactions and of emotion recognition.

## Organization

The folder `/data` contains notebooks for feature extraction and study of the data.
The `/model` folder has two notebooks, one for categorical, and the other, for regression algorithms.

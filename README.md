# Self-Supervised-Learning-for-Medical-Image-Analysis

Self-Supervised Learning for Medical Image Analysis


## About


### Collaborators:
- [Oğuz Ata Çal](https://github.com/OguzAtaCal)
- [Bora Kargı](https://github.com/kargibora)
- [Karahan Sarıtaş](https://github.com/KarahanS)
- [Kıvanç Tezören](https://github.com/kivanctezoren)

## Installation

1. Clone this repository:
```
git clone https://github.com/KarahanS/Self-Supervised-Learning-for-Medical-Image-Analysis.git
```
2. We have used Python 3.10 to obtain our results. Although the requirements should work with higher (and some lower) Python versions, you may want to ensure working Python 3.10 environment for reproducibility. 
 Create a virtual environment and activate it:
```bash
python -m venv env
source env/bin/activate # linux
env\Scripts\activate    # windows
```
3. Install the required packages.
```
pip install -r requirements.txt
```
---

## Usage & Configurations
**TODO:** Explain pretraining/downstream terms, features...

```
python main.py --cfg-path <configuration path>
```

**The user is expected to run `main.py` in either Pretraining or Downstream mode. This is specified in configurations by the inclusion of either the `Training.Pretraining` or `Training.Downstream` field. Both cannot be provided at the same time, and one of them must be provided.**

The [default configuration](src/utils/config/defaults.yaml) provided in the repository shows and explains the expected fields. Whenever a required field cannot be found in a given configuration, the default configuration file can provide its predetermined default value. This is except the `Training.Pretraining` and `Training.Downstream` fields, which are not retreived if not provided by the user. Their subfields, however, are retreived as normal as long as the user provides them in their configuration.

### Adding a new configuration field

New fields can be introduced in configurations and retrieved in code without having to modify anything else.
However, for practicality, the following should be done:
1. The default value for the new field should be provided in the [default configuration](src/utils/config/defaults.yaml).
2. If not covered by existing code, retrieval from the default should be handled in `Config.__init__` before `_sanitize_cfg` is called.
3. The given values should be checked with assertions in `Config._sanitize_cfg`. Paths should end with the delimiter `/`.
4. The desired data type of new values should be casted in `Config._cast_cfg`.

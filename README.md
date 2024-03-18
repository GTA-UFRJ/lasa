# LASA Analysis Tool

The LASA Analysis Tool is a software designed to identify and analyze issues related to Look-Alike, Sound-Alike (LASA) medications in the Brazilian healthcare context. Its primary aim is to enhance patient safety by minimizing medication errors through the facilitated identification of drugs with potential confusion.

## Introduction

Medication errors pose a significant threat to patient safety. The LASA Analysis Tool seeks to mitigate this risk by providing an analysis and reporting tool for LASA medications, promoting better management and prevention of pharmaceutical errors.

## Prerequisites

Ensure you have the following tools installed on your system:

- Python 3.8 or higher
- Required Python libraries as specified in `requirements.txt`

Additionally, the tool depends on the `metaphone-ptbr` library for phonetic comparison of medication names, which can be found at [metaphone-ptbr on GitHub](https://github.com/carlosjordao/metaphone-ptbr).

## Installation

Follow these steps to install and set up the LASA Analysis Tool on your development environment:

1. Clone the repository:
```bash
git clone https://github.com/GTA-UFRJ/lasa.git
```

2. Navigate to the project directory:
```bash
cd lasa
```

3. Install the necessary dependencies:
```bash
pip install -r requirements.txt
```

4. Install the `metaphone-ptbr` library:
```bash
git clone https://github.com/carlosjordao/metaphone-ptbr.git
cd metaphone-ptbr
python setup.py install
```

## Usage

To use the LASA Analysis Tool, follow these instructions:

1. Run the main script with Python:
```bash
python main.py
```

2. Follow the on-screen instructions to input the necessary data or analyze LASA medications.

## How to Contribute

Contributions are always welcome. If you would like to contribute, please:

1. Fork the project.
2. Create a new feature branch (`git checkout -b feature/NewFeature`).
3. Commit your changes (`git commit -am 'Add some NewFeature'`).
4. Push to the branch (`git push origin feature/NewFeature`).
5. Open a Pull Request.


## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License - see the [LICENSE](LICENSE) file for details. This means you are free to share and adapt the material as long as you follow the license terms, which include giving appropriate credit, providing a link to the license, and indicating if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use. You may not use the material for commercial purposes.

For more information on this license, visit [https://creativecommons.org/licenses/by-nc/4.0/](https://creativecommons.org/licenses/by-nc/4.0/).


## Acknowledgments

- The development team thanks all the contributors and the community that contribute to the LASA Analysis Tool.
- Special thanks to healthcare institutions and professionals who share insights and critical data for the project's evolution.

---

This README.md provides a solid foundation that you can expand with specific details of your LASA Analysis Tool project, including unique features, screenshots, and more information on the tool's practical application.

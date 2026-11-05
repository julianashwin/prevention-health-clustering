# PSID documentation not included in the zips

Each data zip already contains that file's **codebook**. Most also include a readme or introduction, and the wealth and DUST zips include their user guides. The zips do **not** include the questionnaires, the question-by-question notes (QxQ), the general user guide, the cross-year variable index or the weights documentation.

Save these into `data/raw/psid/docs/`. All links come from the PSID documentation page: https://psidonline.isr.umich.edu/Guide/documents.aspx

## Essential (about 35 files)

**How the files fit together**
- User Guide 2023: https://psidonline.isr.umich.edu/data/Documentation/UserGuide2023.pdf
- Individual file introduction (also in the zip): https://psidonline.isr.umich.edu/documents/psid/intro/IND2023ER_intro.pdf
- Cross-year variable index: in the Data Center (https://simba.isr.umich.edu/DC/s.aspx, "Variable Search" / cross-year index). It maps each question to its variable name in every wave. I haven't been able to confirm a direct download link, so get it from the Data Center.

**Questionnaires, 1999–2023.** These give the exact wording of the health, employment and health-care-use questions. The detailed health section is in the biennial waves, and 1999 is the first of them.
- https://psidonline.isr.umich.edu/documents/psid/questionnaires/q1999.pdf
- …/q2001.pdf, q2003.pdf, q2005.pdf, q2007.pdf, q2009.pdf, q2011.pdf, q2013.pdf, q2015.pdf, q2017.pdf, q2019.pdf, q2021.pdf, q2023.pdf. These share the same folder as q1999.pdf; change only the year in the name.

**Question-by-question notes (QxQ).** These say who was asked what and give the interviewer instructions. They are only published up to 2015.
- https://psidonline.isr.umich.edu/data/Documentation/Fam/1999/QxQs.pdf
- the same address with 2001, 2003, 2005, 2007, 2009, 2011, 2013 or 2015 in place of 1999

**Weights**
- 2023 longitudinal: https://psidonline.isr.umich.edu/data/weights/long_weight_23.pdf
- 2023 cross-sectional: https://psidonline.isr.umich.edu/data/weights/cross_sec_weights_23.pdf
- 1993–2005 longitudinal weights: https://psidonline.isr.umich.edu/data/weights/Long-weights-doc.pdf

**State codes** (for any regional comparison)
- https://psidonline.isr.umich.edu/data/Documentation/PSIDStateCodes.pdf

## Only if we go back before 1999

- Questionnaires `q68.pdf` … `q97.pdf`, in the same folder as above. The pre-1999 names use two-digit years.
- File descriptions and "tape codes" by year, listed per year on the documentation page.
- The later user guides (2011–2021), if a variable's history needs explaining. The 2023 guide supersedes them for general use.

## Not needed for this project

Child Development Supplement and Transition into Adulthood documentation: hundreds of files about children and young adults. The CDS zips already include their questionnaires.

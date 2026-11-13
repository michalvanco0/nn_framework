import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputDir = "C:\\Users\\User\\diplomovka\\nn_framework\\outputs\\2026-11-13-prehlad-metriky";
const previewMain = "C:\\Users\\User\\diplomovka\\nn_framework\\.artifact-build\\prehlad_main.png";
const previewPlan = "C:\\Users\\User\\diplomovka\\nn_framework\\.artifact-build\\prehlad_plan.png";
await fs.mkdir(outputDir, { recursive: true });

const workbook = Workbook.create();
const main = workbook.worksheets.add("prehlad");

main.getRange("B3:D3").values = [["model", "dataset", "info"]];
main.getRange("A4").values = [["metriky"]];
main.getRange("E4").values = [["klasifikacia"]];
main.getRange("J4").values = [["ROC-AUC"]];
main.getRange("K4").values = [["kalibracia"]];
main.getRange("M4").values = [["skore pravdepodobnosti"]];
main.getRange("O4").values = [["ferovost"]];
main.getRange("Q4").values = [["confidence"]];
main.getRange("E5:I5").values = [[
  "accuracy",
  "balanced accuracy",
  "precision w/m",
  "recall w/m",
  "F1 w/m",
]];
main.getRange("K5:L5").values = [["ECE", "ACE"]];
main.getRange("M5:N5").values = [["Brier score", "log loss"]];
main.getRange("O5:P5").values = [["equalized odds", "equal opportunity"]];
main.getRange("A6").values = [["info, vyznam metriky"]];
main.getRange("D6").values = [["prazdne = nemerane alebo bez jasneho prepojenia na tento beh"]];
main.getRange("E6:Q6").values = [[
  "podiel vsetkych predikcii, ktore boli spravne",
  "priemerny recall napriec triedami; uzitocna pri nerovnovaznych triedach",
  "precision: spravnost oznacenych pozitivnych; poradie je vazeny / makro priemer",
  "recall: kolko skutocnych pozitivnych model nasiel; poradie je vazeny / makro",
  "F1 spaja precision a recall; poradie je vazeny / makro",
  "ako dobre model zoradi pozitivne priklady pri roznom prahu",
  "chyba istoty top-label; nizsie je lepsie",
  "chyba istoty po triedach; nizsie je lepsie",
  "skore kvality pravdepodobnosti; nizsie je lepsie",
  "penalizacia nespravnych pravdepodobnosti; nizsie je lepsie",
  "rozdiel TPR / FPR medzi skupinami pre >50K, v percentnych bodoch",
  "rozdiel TPR medzi skupinami pre >50K, v percentnych bodoch",
  "priemerna istota: vsetky / spravne / nespravne predikcie",
]];

main.getRange("B7:Q9").values = [
  [
    "cnn - ResNet18 (results1)",
    "obrazovy - CIFAR-10",
    "10 obrazovych tried; test n=10000; najlepsia validacna accuracy 95.58% v epoche 135",
    0.947,
    null,
    "94.70% / n/a",
    "94.70% / n/a",
    "94.69% / n/a",
    null,
    null,
    null,
    null,
    "n/a - no demographic groups",
    "n/a - no demographic groups",
    null,
  ],
  [
    "mlp - TabularMLP",
    "tabularny - Adult",
    "prijmova trieda <=50K alebo >50K; pozitivna trieda >50K; test n=16281. Pre >50K: precision 75.66%, recall 55.62%, F1 64.11%. Matica zamen: TN=11747, FP=688, FN=1707, TP=2139. sex, race a age_band boli len na audit, nie ako vstupy modelu; male skupiny race maju n=135 a n=159.",
    0.8528960137583687,
    0.7504172712143709,
    "84.56% / 81.49%",
    "85.29% / 75.04%",
    "84.46% / 77.43%",
    0.9033601456643711,
    null,
    null,
    0.20398651577423077,
    0.3202638488325505,
    "sex TPR/FPR 9.84/6.04pp; race 20.90/5.39pp; age 14.57/6.82pp",
    "sex 9.84pp; race 20.90pp; age 14.57pp",
    null,
  ],
  [
    "cnn - ResNet18 (results)",
    "obrazovy - CIFAR-10",
    "10 obrazovych tried; test n=10000; 7443 spravnych predikcii",
    0.7443,
    null,
    "74.67% / n/a",
    "74.43% / n/a",
    "74.48% / n/a",
    null,
    null,
    null,
    null,
    "n/a - no demographic groups",
    "n/a - no demographic groups",
    "88.41% / 92.99% / 75.08%",
  ],
];

main.mergeCells("A4:A5");
main.mergeCells("E4:I4");
main.mergeCells("J4:J5");
main.mergeCells("K4:L4");
main.mergeCells("M4:N4");
main.mergeCells("O4:P4");
main.mergeCells("Q4:Q5");

main.getRange("E7:F9").format.numberFormat = "0.00%";
main.getRange("J7:J9").format.numberFormat = "0.00%";
main.getRange("K7:L9").format.numberFormat = "0.00%";
main.getRange("M7:N9").format.numberFormat = "0.0000";
main.getRange("A3:Q9").format.font = { name: "Arial", size: 10 };
main.getRange("A3:Q9").format.wrapText = true;
main.getRange("A3:Q9").format.verticalAlignment = "top";
main.getRange("B3:D3").format.font = { name: "Arial", size: 10, bold: true };
main.getRange("A4:Q5").format.font = { name: "Arial", size: 10, bold: true };
main.getRange("A6:Q6").format.font = { name: "Arial", size: 9 };
main.getRange("A1:A12").format.columnWidth = 20;
main.getRange("B1:B12").format.columnWidth = 28;
main.getRange("C1:C12").format.columnWidth = 24;
main.getRange("D1:D12").format.columnWidth = 54;
main.getRange("E1:F12").format.columnWidth = 17;
main.getRange("G1:I12").format.columnWidth = 23;
main.getRange("J1:J12").format.columnWidth = 15;
main.getRange("K1:N12").format.columnWidth = 17;
main.getRange("O1:P12").format.columnWidth = 32;
main.getRange("Q1:Q12").format.columnWidth = 37;
main.getRange("A3:Q9").format.autofitRows();

const plan = workbook.worksheets.add("navrh metrik");
plan.mergeCells("A1:E1");
plan.getRange("A1").values = [["dalsie metriky a vystupy"]];
plan.mergeCells("A3:E3");
plan.getRange("A3").values = [["JSON vystupy bez spolahliveho oznacenia behu"]];
plan.getRange("A4:E8").values = [
  ["report", "vysledky", "co sa da vycitat", "obmedzenie", "stav"],
  [
    "results/calibration.json",
    "ECE 3.54%; ACE 0.74%",
    "ECE: 10 rovnako sirokych binov; ACE: 15 adaptivnych binov po triedach",
    "CIFAR-10 a ResNet18 su uvedene, ale chyba checkpoint/run ID",
    "nepriradene",
  ],
  [
    "results/calibration_1/calibration.json",
    "ACE 2.69%; ECE chyba",
    "10 adaptivnych binov po triedach",
    "run ID a split nie su zaznamenane",
    "nepriradene",
  ],
  [
    "results/fairness_cifar10_superclasses.json",
    "accuracy 89.97%; balanced accuracy 89.97%; macro F1 89.92%; ECE 3.54%; Brier 0.1523; log loss 0.3302",
    "triedy CIFAR su zoskupene na zvierata a vozidla",
    "nie je to demograficka ferovost; chyba checkpoint/run ID",
    "nepriradene",
  ],
  [
    "Adult calibration",
    "v aktualnych JSON suboroch nie je Adult ECE/ACE",
    "bunky v prehlade modelov su preto prazdne",
    "spustit kalibraciu a ulozit nove Adult hodnoty",
    "chyba vystup",
  ],
];

plan.mergeCells("A10:E10");
plan.getRange("A10").values = [["metriky a testy na doplnenie do frameworku"]];
plan.getRange("A11:E21").values = [
  ["oblast", "metrika / test", "co testovat", "co ukladat", "stav / pouzitie"],
  [
    "standard classification",
    "PR-AUC / Average Precision",
    "Adult, pozitivna trieda >50K; uzitocna pri nerovnovaznych triedach",
    "PR-AUC; rozhodovaci prah vyberat na validacnych, nie testovacich datach",
    "zatial nie je v JSON; doplnit",
  ],
  [
    "robustnost - bezne poskodenia obrazu",
    "Gaussian noise, blur, brightness, contrast",
    "Ciste CIFAR-10 obrazy oproti poskodenym pri viacerych urovniach poskodenia",
    "accuracy a F1 pri kazdej urovni; priemerny a najhorsi pokles oproti cistym",
    "zakladny kod v robustness.py; JSON vysledky nie su",
  ],
  [
    "robustnost - adversarialne utoky",
    "FGSM / PGD",
    "CIFAR-10 pri rastucom epsilon; zaznamenat velkost zmeny vstupu",
    "robust accuracy pri kazdom epsilon; uspesnost utoku",
    "navrh",
  ],
  [
    "robustnost - tabularne udaje",
    "chybajuce hodnoty, sum v cislach, zriedkave alebo nezname kategorie",
    "Adult vstupy; menit jeden priznak aj kombinovane stresove scenare",
    "zmena balanced accuracy, macro-F1 a Brier/ECE oproti povodnym udajom",
    "navrh",
  ],
  [
    "posun v rozdeleni dat",
    "ina populacia, obdobie alebo zdroj dat",
    "CIFAR z ineho zdroja; Adult oproti ACSIncome",
    "klasifikacia, kalibracia a rozdiely skupin na zmenenych datach",
    "navrh; treba externy dataset",
  ],
  [
    "OOD detekcia",
    "max softmax probability, entropia, energy score, vzdialenost priznakov",
    "zname oproti neznamym vstupom; napr. CIFAR-10 vs SVHN/Textures, Adult vs ACSIncome",
    "AUROC, AUPR, FPR@95TPR a zvoleny prah",
    "navrh; treba OOD data",
  ],
  [
    "selektivna predikcia",
    "risk-coverage curve / AURC",
    "model odmietne predikcie s nizkou istotou",
    "coverage a chyba na prijatych pripadoch pri zmene prahu; AURC",
    "zatial nie je ulozeny JSON vystup",
  ],
  [
    "rozsirenie ferovosti",
    "prieniky skupin a bootstrap intervaly",
    "Adult sex x race alebo sex x age; ukazat pocty v skupinach",
    "equal opportunity/equalized odds rozdiely s 95% intervalom spolahlivosti",
    "zakladne rozdiely su; rozsirenie je navrh",
  ],
  [
    "opakovatelnost",
    "viac nahodnych seedov pri treningu",
    "kazdy model spustit 3-5 krat s rovnakym splitom a nastavenim",
    "priemer +/- smerodajna odchylka alebo 95% CI pre accuracy, F1, kalibraciu a ferovost",
    "navrh",
  ],
  [
    "konformalna predikcia",
    "mnoziny moznych tried",
    "Adult a CIFAR pravdepodobnosti s oddelenym kalibracnym splitom",
    "skutocne pokrytie oproti priemernej velkosti mnoziny",
    "volitelne do buducna",
  ],
];

plan.getRange("A1:E21").format.font = { name: "Arial", size: 10 };
plan.getRange("A1:E21").format.wrapText = true;
plan.getRange("A1:E21").format.verticalAlignment = "top";
plan.getRange("A1:E1").format.font = { name: "Arial", size: 14, bold: true };
plan.getRange("A3:E3").format.font = { name: "Arial", size: 11, bold: true };
plan.getRange("A10:E10").format.font = { name: "Arial", size: 11, bold: true };
plan.getRange("A4:E4").format.font = { name: "Arial", size: 10, bold: true };
plan.getRange("A11:E11").format.font = { name: "Arial", size: 10, bold: true };
plan.getRange("A1:A21").format.columnWidth = 28;
plan.getRange("B1:B21").format.columnWidth = 38;
plan.getRange("C1:C21").format.columnWidth = 54;
plan.getRange("D1:D21").format.columnWidth = 56;
plan.getRange("E1:E21").format.columnWidth = 34;
plan.getRange("A1:E21").format.autofitRows();

workbook.recalculate();
const summary = await workbook.inspect({ kind: "workbook,sheet,table", maxChars: 2600 });
const mainView = await workbook.inspect({ kind: "region", sheetId: main.name, range: "A3:Q9", maxChars: 3200 });
const planView = await workbook.inspect({ kind: "region", sheetId: plan.name, range: "A1:E21", maxChars: 5000 });
console.log("SUMMARY\n" + summary.ndjson);
console.log("MAIN SHEET\n" + mainView.ndjson);
console.log("PLAN SHEET\n" + planView.ndjson);

for (const [sheetName, path] of [[main.name, previewMain], [plan.name, previewPlan]]) {
  const rendered = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(path, new Uint8Array(await rendered.arrayBuffer()));
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(`${outputDir}\\prehlad.xlsx`);
console.log(`SAVED ${outputDir}\\prehlad.xlsx`);

import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputDir = "C:\\Users\\User\\diplomovka\\nn_framework\\outputs\\2026-11-12-simple-prehlad";
const previewPath = "C:\\Users\\User\\diplomovka\\nn_framework\\.artifact-build\\preview.png";
await fs.mkdir(outputDir, { recursive: true });

const workbook = Workbook.create();
const sheet = workbook.worksheets.add("Prehľad");

sheet.getRange("A1:C1").merge();
sheet.getRange("A1").values = [["Prehľad modelov a metrík"]];
sheet.getRange("A3:C3").merge();
sheet.getRange("A3").values = [["Modely, ktoré sme trénovali a testovali"]];
sheet.getRange("A4:C6").values = [
  ["Model", "Dataset a úloha", "Kde sa hodí a čo už máme"],
  [
    "ResNet18 (2 behy)",
    "CIFAR-10\nTriedenie obrázkov do 10 tried",
    "Študijný benchmark na triedenie podobných obrázkov. Test accuracy: 74,4 % a 94,7 % v dvoch samostatných behoch.",
  ],
  [
    "TabularMLP",
    "Adult\nOdhad príjmovej skupiny: ≤50K alebo >50K",
    "Ukážka klasifikácie tabuľkových údajov, nie rozhodovanie o príjme konkrétnych ľudí. Test accuracy: 85,3 %. Pri >50K model zachytil 55,6 % skutočných prípadov. Pohlavie, rasa a veková skupina slúžili na audit, nie ako vstupy modelu.",
  ],
];

sheet.getRange("A8:C8").merge();
sheet.getRange("A8").values = [["Metriky a kontroly, ktoré sme už použili"]];
sheet.getRange("A9:C15").values = [
  ["Metrika / kontrola", "Kde ju máme", "Čo nám jednoducho povie"],
  [
    "Accuracy",
    "Adult a oba behy CIFAR-10",
    "Podiel správnych odpovedí. Ľahko sa číta, ale môže skryť horší výsledok na menej častej triede.",
  ],
  [
    "Balanced accuracy",
    "Adult",
    "Priemer zachytených prípadov naprieč triedami; pomáha, keď je jedna trieda početnejšia než druhá.",
  ],
  [
    "Precision / recall / F1",
    "Adult a CIFAR-10 majú vážené hodnoty; macro-F1 máme aj pre Adult.",
    "Precision: koľko pozitívnych odhadov bolo správnych. Recall: koľko skutočných pozitívnych prípadov model našiel. F1 spája obe.",
  ],
  [
    "ROC-AUC",
    "Adult (>50K)",
    "Ukazuje, ako dobre model zoradí prípady >50K pred prípady ≤50K pri rôznych rozhodovacích prahoch.",
  ],
  [
    "ECE / ACE (kalibrácia)",
    "Adult; kalibračný report CIFAR-10 má nejasnú väzbu na konkrétny beh.",
    "Ukazuje, či istota modelu zodpovedá jeho úspešnosti. Porovnávame ich len pri rovnakom nastavení výpočtu.",
  ],
  [
    "Skupinový audit",
    "Adult: pohlavie, rasa a veková skupina. CIFAR-10: zvieratá/vozidlá.",
    "Ukazuje rozdiely medzi skupinami. CIFAR skupiny sú podľa typu obrázkov, nie podľa ľudí; audit nedáva jedno celkové hodnotenie férovosti.",
  ],
];

sheet.getRange("A17:C17").merge();
sheet.getRange("A17").values = [["Poznámka: 74,4 % a 94,7 % pri CIFAR-10 sú výsledky dvoch samostatných behov ResNet18."]];

sheet.getRange("A1:C17").format.font = { name: "Arial", size: 11 };
sheet.getRange("A1:C17").format.wrapText = true;
sheet.getRange("A1:C17").format.verticalAlignment = "top";
sheet.getRange("A1:C1").format.font = { name: "Arial", size: 16, bold: true };
sheet.getRange("A3:C3").format.font = { name: "Arial", size: 12, bold: true };
sheet.getRange("A8:C8").format.font = { name: "Arial", size: 12, bold: true };
sheet.getRange("A4:C4").format.font = { name: "Arial", size: 11, bold: true };
sheet.getRange("A9:C9").format.font = { name: "Arial", size: 11, bold: true };
sheet.getRange("A17:C17").format.font = { name: "Arial", size: 10, italic: true };
sheet.getRange("A1:A17").format.columnWidth = 27;
sheet.getRange("B1:B17").format.columnWidth = 49;
sheet.getRange("C1:C17").format.columnWidth = 78;
sheet.getRange("A1:C17").format.autofitRows();

workbook.recalculate();
const summary = await workbook.inspect({ kind: "workbook,sheet,table", maxChars: 2500 });
const contents = await workbook.inspect({ kind: "region", sheetId: sheet.name, range: "A1:C17", maxChars: 7000 });
console.log("WORKBOOK SUMMARY\n" + summary.ndjson);
console.log("WORKSHEET CONTENTS\n" + contents.ndjson);

const preview = await workbook.render({ sheetName: sheet.name, autoCrop: "all", scale: 1, format: "png" });
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(`${outputDir}\\prehlad_modelov_a_metrik.xlsx`);
console.log(`SAVED ${outputDir}\\prehlad_modelov_a_metrik.xlsx`);

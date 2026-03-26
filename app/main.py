import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool


app = FastAPI()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DATASETS_FILE = Path(__file__).resolve().parent / "datasets.json"

current_df: pd.DataFrame | None = None
current_file_name: str | None = None
current_dataset_id: str | None = None
upload_status = {"status": "idle", "progress": 0}


class CountRequest(BaseModel):
    column: str


class GroupRequest(BaseModel):
    column: str | None = None
    group_by: str | None = None
    breakdown: list[str] | None = None
    filters: dict[str, object] | None = None


class UniqueValuesRequest(BaseModel):
    column: str


class AnalyzeRequest(BaseModel):
    breakdown: list[str] | None = None
    filters: dict[str, object] | None = None


class NormalizeColumnRequest(BaseModel):
    column: str


class AISuggestRequest(BaseModel):
    message: str | None = None
    filters: dict[str, object] | None = None
    breakdown: list[str] | None = None


class AIQueryRequest(BaseModel):
    message: str
    summary: dict[str, object] | None = None


def save_file(file_path: Path, content: bytes):
    file_path.write_bytes(content)


def load_datasets() -> list[dict[str, object]]:
    if not DATASETS_FILE.exists():
        return []
    try:
        return json.loads(DATASETS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_datasets(datasets: list[dict[str, object]]):
    DATASETS_FILE.write_text(
        json.dumps(datasets, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_tags(tags_text: str | None) -> list[str]:
    if not tags_text:
        return []
    return [tag.strip() for tag in tags_text.split(",") if tag.strip()]


def read_table_from_path(file_path: Path):
    filename = file_path.name.lower()
    if filename.endswith(".xlsx"):
        return pd.read_excel(file_path)
    try:
        return pd.read_csv(
            file_path,
            encoding="utf-8",
            on_bad_lines="skip",
            sep=None,
            engine="python",
        )
    except Exception:
        return pd.read_csv(
            file_path,
            encoding="utf-8",
            on_bad_lines="skip",
            sep=";",
            engine="python",
        )


def get_dataset(dataset_id: str) -> dict[str, object] | None:
    for dataset in load_datasets():
        if dataset["id"] == dataset_id:
            return dataset
    return None


def load_current_dataset(dataset_id: str):
    global current_df, current_file_name, current_dataset_id

    dataset = get_dataset(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_path = Path(str(dataset["file_path"]))
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found")

    current_df = read_table_from_path(file_path)
    current_file_name = str(dataset["name"])
    current_dataset_id = dataset_id
    return dataset


def create_dataset_record(name: str, file_path: Path, tags: list[str]) -> dict[str, object]:
    datasets = load_datasets()
    dataset = {
        "id": uuid.uuid4().hex,
        "name": name,
        "file_path": str(file_path),
        "created_at": datetime.utcnow().isoformat(),
        "tags": tags,
    }
    datasets.append(dataset)
    save_datasets(datasets)
    return dataset


def ensure_data_dir() -> Path:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        test_path = DATA_DIR / ".write_test"
        test_path.write_text("ok", encoding="utf-8")
        test_path.unlink(missing_ok=True)
        return DATA_DIR
    except Exception:
        tmp_dir = Path("/tmp") / "table-analysis-tool-data"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        return tmp_dir


def read_table(upload_file: UploadFile):
    filename = (upload_file.filename or "").lower()

    if filename.endswith(".xlsx"):
        upload_file.file.seek(0)
        return pd.read_excel(upload_file.file)

    upload_file.file.seek(0)
    try:
        return pd.read_csv(
            upload_file.file,
            encoding="utf-8",
            on_bad_lines="skip",
            sep=None,
            engine="python",
        )
    except Exception:
        upload_file.file.seek(0)
        return pd.read_csv(
            upload_file.file,
            encoding="utf-8",
            on_bad_lines="skip",
            sep=";",
            engine="python",
        )


def get_column_data(df: pd.DataFrame, column: str):
    if column.isdigit():
        return df.iloc[:, int(column)]
    return df[column]


def get_column_key(df: pd.DataFrame, column: str):
    if column.isdigit():
        return df.columns[int(column)]
    return column


def build_group_tree(df: pd.DataFrame, breakdown: list[str]):
    if not breakdown:
        return {"count": int(len(df))}

    current_column = breakdown[0]
    col_data = get_column_data(df, current_column).dropna().astype(str)
    col_data = col_data.loc[col_data.str.strip() != ""]
    filtered_df = df.loc[col_data.index]
    grouped = filtered_df.groupby(col_data, sort=False)

    result = []
    for group, group_df in grouped:
        item = {"group": str(group), "count": int(len(group_df))}
        if len(breakdown) > 1:
            item["subgroups"] = build_group_tree(group_df, breakdown[1:])
        result.append(item)
    return result


def sort_groups_by_count(items: list[dict[str, object]]):
    return sorted(items, key=lambda item: item["count"], reverse=True)


def sort_group_tree(items):
    if not isinstance(items, list):
        return items

    sorted_items = sort_groups_by_count(items)
    for item in sorted_items:
        if "subgroups" in item:
            item["subgroups"] = sort_group_tree(item["subgroups"])
    return sorted_items


def apply_filters(df: pd.DataFrame, filters: dict[str, object] | None):
    if filters:
        for key, value in filters.items():
            if key not in df.columns:
                raise HTTPException(status_code=400, detail=f"Filter column not found: {key}")
            if isinstance(value, list):
                df = df[df[key].isin(value)]
            elif isinstance(value, str):
                df = df[df[key].astype(str).str.contains(value, case=False, na=False)]
            else:
                df = df[df[key] == value]
    return df


def dataframe_page_to_records(df: pd.DataFrame, page: int, page_size: int):
    start = max(page - 1, 0) * page_size
    end = start + page_size
    page_df = df.iloc[start:end].astype(object).where(pd.notna(df.iloc[start:end]), None)
    return page_df.to_dict(orient="records")


@app.get("/", response_class=HTMLResponse)
def datasets_index():
    datasets = load_datasets()
    dataset_items = "".join(
        f"""
        <div style="border:1px solid #d1d5db;border-radius:8px;padding:12px;margin-top:12px;">
            <div><strong>{dataset["name"]}</strong></div>
            <div style="color:#6b7280;font-size:14px;margin-top:4px;">{dataset["created_at"]}</div>
            <div style="color:#374151;font-size:14px;margin-top:4px;">Tags: {", ".join(dataset.get("tags", [])) or "-"}</div>
            <div style="margin-top:8px;">
                <a href="/app?dataset_id={dataset["id"]}">Open</a>
            </div>
        </div>
        """
        for dataset in reversed(datasets)
    )
    return HTMLResponse(
        f"""<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>Datasets</title>
</head>
<body style="font-family:Arial,sans-serif;background:#f3f4f6;margin:0;padding:24px;">
    <div style="max-width:960px;margin:0 auto;">
        <h1>Datasets</h1>
        <form action="/datasets/upload" method="post" enctype="multipart/form-data" style="background:#fff;border:1px solid #d1d5db;border-radius:8px;padding:16px;">
            <div>
                <input type="file" name="file" required />
            </div>
            <div style="margin-top:12px;">
                <input type="text" name="tags" placeholder="tags, comma, separated" style="width:100%;padding:8px 10px;border:1px solid #d1d5db;border-radius:6px;" />
            </div>
            <div style="margin-top:12px;">
                <button type="submit">Upload</button>
            </div>
        </form>
        <div style="margin-top:24px;">
            {dataset_items or '<div style="background:#fff;border:1px solid #d1d5db;border-radius:8px;padding:16px;">No datasets yet</div>'}
        </div>
    </div>
</body>
</html>"""
    )


@app.get("/app", response_class=HTMLResponse)
def index(dataset_id: str | None = Query(default=None)):
    print("HTML RESPONSE FROM MAIN.PY")
    initial_dataset = None
    if dataset_id:
        initial_dataset = load_current_dataset(dataset_id)
    return HTMLResponse(
        """<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>Table Analysis Tool</title>
    <meta http-equiv="Cache-Control" content="no-store, no-cache, must-revalidate, max-age=0">
    <meta http-equiv="Pragma" content="no-cache">
    <meta http-equiv="Expires" content="0">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/ag-grid-community@31.3.4/styles/ag-grid.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/ag-grid-community@31.3.4/styles/ag-theme-alpine.css">
    <script src="https://cdn.jsdelivr.net/npm/ag-grid-community@31.3.4/dist/ag-grid-community.min.js"></script>
    <style>
    * { box-sizing: border-box; }
    html, body {
        margin: 0;
        padding: 0;
        min-height: 100%;
        font-family: Arial, sans-serif;
        background: #f3f4f6;
        color: #111827;
    }
    body { overflow: auto; }
    .app {
        display: grid;
        grid-template-columns: 320px minmax(0, 1fr);
        grid-template-rows: auto auto auto auto;
        gap: 12px;
        min-height: 100vh;
        padding: 12px;
    }
    .panel {
        background: #fff;
        border: 1px solid #dde3ea;
        border-radius: 4px;
        padding: 12px;
    }
    .panel h2 {
        margin: 0 0 10px;
        font-size: 16px;
        font-weight: 600;
    }
    .header-panel {
        grid-column: 1 / 3;
        display: grid;
        grid-template-columns: minmax(0, 1fr) 260px;
        gap: 12px;
        align-items: start;
        padding: 6px 8px;
    }
    .sidebar {
        grid-column: 1;
        grid-row: 2 / 5;
        display: flex;
        flex-direction: column;
        gap: 12px;
        min-width: 0;
    }
    .result-panel {
        grid-column: 2;
        grid-row: 2;
        min-width: 0;
        display: flex;
        flex-direction: column;
        gap: 12px;
        min-height: 420px;
        border-color: #cfd8e3;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
    }
    .ai-panel {
        grid-column: 2;
        grid-row: 3;
        min-width: 0;
        padding: 10px 12px;
        background: #f8fafc;
        border-color: #e5eaf0;
    }
    .preview-panel {
        grid-column: 2;
        grid-row: 4;
        padding: 10px 12px;
        background: #fbfcfd;
        border-color: #e7ebf0;
    }
    .header-actions {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }
    .upload-status {
        margin-top: 6px;
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
    }
    .upload-progress {
        width: 220px;
        height: 10px;
        border: 1px solid #d1d5db;
        border-radius: 999px;
        background: #f3f4f6;
        overflow: hidden;
    }
    .upload-progress-bar {
        height: 100%;
        width: 0%;
        background: #2563eb;
    }
    .dataset-info {
        min-height: 0;
        color: #374151;
        background: #f9fafb;
        border: 1px dashed #d7dde5;
        border-radius: 4px;
        padding: 8px 10px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 4px;
    }
    .dataset-line {
        font-size: 12px;
        color: #4b5563;
    }
    .header-panel > div > h2 {
        margin-bottom: 2px;
        font-size: 14px;
    }
    .placeholder, .muted-text {
        color: #6b7280;
        font-size: 13px;
    }
    .control-block + .control-block { margin-top: 10px; }
    label {
        display: block;
        font-size: 13px;
        margin-bottom: 5px;
        color: #374151;
    }
    select, textarea, input[type="text"] {
        width: 100%;
        max-width: 100%;
        min-width: 0;
        padding: 7px 9px;
        border: 1px solid #d1d5db;
        border-radius: 4px;
        background: #fff;
        font: inherit;
    }
    select {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    button {
        padding: 7px 10px;
        border: 1px solid #cbd5e1;
        border-radius: 4px;
        background: #f8fafc;
        cursor: pointer;
        font: inherit;
    }
    button:hover { background: #eef2f7; }
    button:disabled {
        opacity: 0.6;
        cursor: default;
    }
    .button-row {
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
    }
    .filter-list, .breakdown-list {
        display: flex;
        flex-direction: column;
        gap: 6px;
        margin-top: 10px;
    }
    .filter-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        padding: 7px 9px;
        border: 1px solid #d1d5db;
        border-radius: 4px;
        background: #f9fafb;
    }
    .filter-label {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .breakdown-hint {
        color: #6b7280;
        font-size: 14px;
        margin-bottom: 12px;
    }
    .breakdown-item {
        display: grid;
        grid-template-columns: auto minmax(0, 1fr) auto auto auto;
        align-items: center;
        gap: 8px;
        padding: 8px 10px;
        border: 1px solid #d1d5db;
        border-radius: 6px;
        background: #f9fafb;
    }
    .breakdown-label {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .subpanel {
        margin-top: 16px;
        padding-top: 16px;
        border-top: 1px solid #e5e7eb;
    }
    .chat-input {
        width: 100%;
        min-height: 72px;
        resize: vertical;
    }
    .chat-messages {
        margin-top: 12px;
        max-height: 220px;
        overflow: auto;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        background: #f9fafb;
        padding: 12px;
    }
    .chat-message + .chat-message { margin-top: 8px; }
    .result-toolbar {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }
    #ag-grid {
        width: 100%;
        height: 360px;
        min-height: 360px;
    }
    #result-json-view {
        display: none;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        background: #fff;
    }
    #result-json {
        margin: 0;
        padding: 12px;
        white-space: pre-wrap;
        word-break: break-word;
        overflow: auto;
        max-height: 360px;
    }
    #dataset-grid {
        width: 100%;
        height: 50vh;
        min-height: 420px;
    }
    .dataset-toolbar {
        display: flex;
        gap: 12px;
        align-items: center;
        flex-wrap: wrap;
    }
    .ag-theme-alpine {
        --ag-font-size: 13px;
        --ag-grid-size: 6px;
        --ag-row-hover-color: #f8fafc;
    }
    @media (max-width: 1100px) {
        .app {
            grid-template-columns: 1fr;
            grid-template-rows: auto auto auto auto;
        }
        .header-panel, .sidebar, .result-panel, .ai-panel, .preview-panel {
            grid-column: 1;
            grid-row: auto;
        }
        .header-panel { grid-template-columns: 1fr; }
        #ag-grid { height: 520px; min-height: 520px; }
        #dataset-grid { height: 320px; min-height: 320px; }
    }
    </style>
</head>
<body>
<div class="app">
    <div class="panel header-panel">
        <div>
            <h2>Header</h2>
            <div class="header-actions">
                <input type="file" id="file" />
                <button onclick="upload()">Upload</button>
            </div>
            <div class="upload-status">
                <div class="upload-progress">
                    <div id="upload-progress-bar" class="upload-progress-bar"></div>
                </div>
                <div id="upload-status-text" class="muted-text">idle</div>
            </div>
        </div>
        <div>
            <h2>Dataset Info</h2>
            <div class="dataset-info" id="dataset-info">
                <div class="dataset-line">No dataset loaded</div>
            </div>
        </div>
    </div>

    <div class="sidebar">
        <div class="panel filters-panel">
            <h2>Filters</h2>
            <div class="control-block">
                <label for="columns">Columns</label>
                <select id="columns"></select>
            </div>
            <div class="control-block button-row">
                <button id="normalize-button" onclick="normalizeSelectedColumn()">Normalize column</button>
                <span id="normalize-status" class="muted-text"></span>
            </div>
            <div class="control-block">
                <label for="values">Values</label>
                <select id="values"></select>
            </div>
            <div class="control-block button-row">
                <button onclick="addSelectedFilter()">Add filter</button>
                <button onclick="clearFilters()">Clear filters</button>
            </div>
            <div id="filters" class="filter-list"></div>
        </div>

        <div class="panel breakdown-panel">
            <h2>Breakdown</h2>
            <div class="breakdown-hint">Порядок определяет вложенность группировки</div>
            <div class="control-block">
                <label for="breakdown-column">Breakdown column</label>
                <select id="breakdown-column"></select>
            </div>
            <div class="control-block button-row">
                <button onclick="addBreakdown()">+ Add</button>
                <button onclick="clearBreakdown()">Clear breakdown</button>
            </div>
            <div id="breakdown-list" class="breakdown-list"></div>
        </div>
    </div>

    <div class="panel result-panel">
        <h2>Results</h2>
        <div class="result-toolbar">
            <button onclick="setResultMode('table')">Table</button>
            <button onclick="setResultMode('json')">JSON</button>
            <button onclick="group()">Refresh</button>
            <button onclick="openCharts()">Build chart</button>
        </div>
        <div id="ag-grid" class="ag-theme-alpine"></div>
        <div id="result-json-view">
            <pre id="result-json"></pre>
        </div>
    </div>

    <div class="panel ai-panel">
        <h2>AI Assistant</h2>
        <textarea id="ai-input" class="chat-input" placeholder="Ask AI..."></textarea>
        <div style="margin-top: 12px;">
            <button id="ai-button" onclick="sendAIMessage()">Send</button>
        </div>
        <div id="ai-status" class="muted-text" style="margin-top: 12px;"></div>
        <div id="ai-messages" class="chat-messages"></div>
    </div>

    <div class="panel preview-panel">
        <h2>Dataset Viewer</h2>
        <div class="dataset-toolbar">
            <button onclick="prevDatasetPage()">Prev</button>
            <div id="dataset-page" class="muted-text">Page 1</div>
            <button onclick="nextDatasetPage()">Next</button>
            <div id="dataset-total" class="muted-text">Rows: 0</div>
        </div>
        <div id="dataset-grid" class="ag-theme-alpine" style="margin-top: 12px;"></div>
    </div>
</div>

<script>
let activeFilters = {};
let activeColumn = null;
let activeBreakdown = [];
let resultMode = 'table';
let latestResult = null;
let aiMessages = [];
let gridApi = null;
let gridColumnApi = null;
let gridInitialized = false;
let datasetGridApi = null;
let datasetGridColumnApi = null;
let datasetGridInitialized = false;
let datasetPage = 1;
let datasetPageSize = 100;
let datasetTotalRows = 0;
let statusPollInterval = null;

function initGrid() {
    const gridElement = document.getElementById('ag-grid');
    if (!gridElement || gridInitialized) {
        return;
    }

    const gridOptions = {
        columnDefs: [
            {
                field: 'count',
                headerName: 'Count',
                sortable: true,
                resizable: true,
                filter: true,
                flex: 1,
                minWidth: 120,
            }
        ],
        rowData: [],
        defaultColDef: {
            sortable: true,
            resizable: true,
            filter: true,
            flex: 1,
            minWidth: 140,
        },
        animateRows: false,
        onGridReady: (params) => {
            gridApi = params.api;
            gridColumnApi = params.columnApi;
        },
        overlayNoRowsTemplate: '<span style="padding:12px; display:inline-block; color:#6b7280;">No data to display</span>',
    };

    new agGrid.Grid(gridElement, gridOptions);
    gridInitialized = true;
}

function initDatasetGrid() {
    const gridElement = document.getElementById('dataset-grid');
    if (!gridElement || datasetGridInitialized) {
        return;
    }

    const gridOptions = {
        columnDefs: [],
        rowData: [],
        defaultColDef: {
            sortable: true,
            resizable: true,
            filter: true,
            minWidth: 140,
            flex: 1,
            tooltipValueGetter: (params) => params.value,
            cellStyle: {
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
            },
        },
        animateRows: false,
        onGridReady: (params) => {
            datasetGridApi = params.api;
            datasetGridColumnApi = params.columnApi;
        },
        overlayNoRowsTemplate: '<span style="padding:12px; display:inline-block; color:#6b7280;">No data to display</span>',
    };

    new agGrid.Grid(gridElement, gridOptions);
    datasetGridInitialized = true;
}

function updateDatasetPagination() {
    const pageLabel = document.getElementById('dataset-page');
    const totalLabel = document.getElementById('dataset-total');
    pageLabel.innerText = `Page ${datasetPage}`;
    totalLabel.innerText = `Rows: ${datasetTotalRows}`;
}

async function loadDatasetPage(page) {
    initDatasetGrid();

    const safePage = Math.max(page, 1);
    const res = await fetch(`/dataset_rows?page=${safePage}&page_size=${datasetPageSize}`);
    const data = await res.json();

    datasetPage = safePage;
    datasetTotalRows = data.total_rows || 0;
    updateDatasetPagination();

    const columnDefs = (data.columns || []).map(column => ({
        field: column,
        headerName: column,
        sortable: true,
        resizable: true,
        filter: true,
        flex: 1,
        minWidth: 140,
        tooltipField: column,
        cellStyle: {
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
        },
    }));

    if (!datasetGridApi) {
        setTimeout(() => loadDatasetPage(safePage), 0);
        return;
    }

    datasetGridApi.setColumnDefs(columnDefs);
    datasetGridApi.setRowData(data.rows || []);
}

async function prevDatasetPage() {
    if (datasetPage <= 1) {
        return;
    }
    await loadDatasetPage(datasetPage - 1);
}

async function nextDatasetPage() {
    if (datasetPage * datasetPageSize >= datasetTotalRows) {
        return;
    }
    await loadDatasetPage(datasetPage + 1);
}

function updateDatasetInfo(fileName, rowCount, columnCount) {
    const info = document.getElementById('dataset-info');
    if (!fileName) {
        info.innerHTML = '<div class="dataset-line">No dataset loaded</div>';
        return;
    }

    info.innerHTML = `
        <div class="dataset-line"><strong>File:</strong> ${fileName}</div>
        <div class="dataset-line"><strong>Rows:</strong> ${rowCount}</div>
        <div class="dataset-line"><strong>Columns:</strong> ${columnCount}</div>
    `;
}

async function initializeLoadedDataset() {
    const currentDatasetRes = await fetch('/current_dataset');
    const currentDataset = await currentDatasetRes.json();

    if (!currentDataset.id && !currentDataset.name) {
        return;
    }

    const columnsRes = await fetch('/columns');
    const data = await columnsRes.json();
    const select = document.getElementById('columns');
    const breakdownSelect = document.getElementById('breakdown-column');
    select.innerHTML = '';
    breakdownSelect.innerHTML = '';

    data.columns.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c.name;
        opt.text = c.name;
        select.appendChild(opt);

        const breakdownOpt = document.createElement('option');
        breakdownOpt.value = c.name;
        breakdownOpt.text = c.name;
        breakdownSelect.appendChild(breakdownOpt);
    });

    select.onchange = async () => {
        activeColumn = select.value || null;
        await loadUniqueValues(select.value);
    };

    activeColumn = select.value || null;
    updateDatasetInfo(
        currentDataset.name,
        currentDataset.row_count || 0,
        currentDataset.column_count || 0
    );

    if (select.value) {
        await loadUniqueValues(select.value);
    }

    datasetPage = 1;
    await loadDatasetPage(1);
    await group();
}

function updateUploadStatus(statusData) {
    const progressBar = document.getElementById('upload-progress-bar');
    const statusText = document.getElementById('upload-status-text');
    const progress = Math.max(0, Math.min(100, Number(statusData?.progress || 0)));
    const status = statusData?.status || 'idle';
    const error = statusData?.error ? ` ${statusData.error}` : '';

    progressBar.style.width = `${progress}%`;
    statusText.innerText = `${progress}% ${status}${error}`;
}

async function pollStatus() {
    try {
        const res = await fetch('/status');
        const statusData = await res.json();
        updateUploadStatus(statusData);

        if (statusData.status === 'done' || statusData.status === 'error') {
            stopStatusPolling();
        }
    } catch (error) {
        updateUploadStatus({ status: 'error', progress: 0, error: String(error) });
        stopStatusPolling();
    }
}

function startStatusPolling() {
    stopStatusPolling();
    pollStatus();
    statusPollInterval = setInterval(pollStatus, 700);
}

function stopStatusPolling() {
    if (statusPollInterval) {
        clearInterval(statusPollInterval);
        statusPollInterval = null;
    }
}

async function upload() {
    const fileInput = document.getElementById('file');
    const file = fileInput.files[0];
    if (!file) {
        return;
    }

    const formData = new FormData();
    formData.append("file", file);
    startStatusPolling();

    let rows;
    try {
        const uploadRes = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        if (!uploadRes.ok) {
            const errorText = await uploadRes.text();
            updateUploadStatus({
                status: 'error',
                progress: 0,
                error: errorText || 'Upload failed',
            });
            stopStatusPolling();
            return;
        }

        rows = await uploadRes.json();

        if (rows?.error) {
            updateUploadStatus({
                status: 'error',
                progress: 0,
                error: rows.error,
            });
            stopStatusPolling();
            return;
        }
    } catch (error) {
        updateUploadStatus({ status: 'error', progress: 0, error: String(error) });
        stopStatusPolling();
        return;
    }

    const columnsRes = await fetch('/columns');
    const data = await columnsRes.json();

    const select = document.getElementById('columns');
    const breakdownSelect = document.getElementById('breakdown-column');
    select.innerHTML = '';
    breakdownSelect.innerHTML = '';

    data.columns.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c.name;
        opt.text = c.name;
        select.appendChild(opt);

        const breakdownOpt = document.createElement('option');
        breakdownOpt.value = c.name;
        breakdownOpt.text = c.name;
        breakdownSelect.appendChild(breakdownOpt);
    });

    select.onchange = async () => {
        activeColumn = select.value || null;
        await loadUniqueValues(select.value);
    };

    activeFilters = {};
    activeColumn = select.value || null;
    activeBreakdown = [];
    renderFilters();
    renderBreakdown();
    updateDatasetInfo(
        file.name,
        Array.isArray(rows) ? rows.length : 0,
        data.columns.length
    );

    if (select.value) {
        await loadUniqueValues(select.value);
    }

    renderAIMessages();
    datasetPage = 1;
    await loadDatasetPage(1);
    await group();
    await pollStatus();
}

function renderFilters() {
    const filters = document.getElementById('filters');
    const entries = Object.entries(activeFilters);
    filters.innerHTML = '';

    if (!entries.length) {
        filters.innerHTML = '<div class="muted-text">No active filters</div>';
        return;
    }

    entries.forEach(([key, value]) => {
        const item = document.createElement('div');
        item.className = 'filter-item';

        const label = document.createElement('span');
        label.className = 'filter-label';
        label.innerText = `${key} = ${value}`;

        const removeButton = document.createElement('button');
        removeButton.innerText = 'x';
        removeButton.onclick = () => removeFilter(key);

        item.appendChild(label);
        item.appendChild(removeButton);
        filters.appendChild(item);
    });
}

function renderBreakdown() {
    const list = document.getElementById('breakdown-list');
    list.innerHTML = '';

    if (!activeBreakdown.length) {
        list.innerHTML = '<div class="muted-text">No breakdown selected</div>';
        return;
    }

    activeBreakdown.forEach((column, index) => {
        const item = document.createElement('div');
        item.className = 'breakdown-item';

        const position = document.createElement('span');
        position.innerText = `${index + 1}.`;

        const label = document.createElement('span');
        label.className = 'breakdown-label';
        label.innerText = column;

        const upButton = document.createElement('button');
        upButton.innerText = '↑';
        upButton.disabled = index === 0;
        upButton.onclick = () => moveBreakdownUp(index);

        const downButton = document.createElement('button');
        downButton.innerText = '↓';
        downButton.disabled = index === activeBreakdown.length - 1;
        downButton.onclick = () => moveBreakdownDown(index);

        const removeButton = document.createElement('button');
        removeButton.innerText = 'x';
        removeButton.onclick = () => removeBreakdown(index);

        item.appendChild(position);
        item.appendChild(label);
        item.appendChild(upButton);
        item.appendChild(downButton);
        item.appendChild(removeButton);
        list.appendChild(item);
    });
}

function setResultMode(mode) {
    resultMode = mode;
    const grid = document.getElementById('ag-grid');
    const jsonView = document.getElementById('result-json-view');

    if (mode === 'json') {
        grid.style.display = 'none';
        jsonView.style.display = 'block';
    } else {
        grid.style.display = 'block';
        jsonView.style.display = 'none';
    }
}

function flattenResult(items, breakdown, path = {}) {
    if (!Array.isArray(items)) {
        if (items && typeof items.count === 'number') {
            return [{ ...path, count: items.count }];
        }
        return [];
    }

    const levelIndex = Object.keys(path).length;
    const currentKey = breakdown[levelIndex] || `Level ${levelIndex + 1}`;
    let rows = [];

    items.forEach(item => {
        const nextPath = { ...path, [currentKey]: item.group };
        if (Array.isArray(item.subgroups) && item.subgroups.length) {
            rows.push(...flattenResult(item.subgroups, breakdown, nextPath));
        } else {
            rows.push({ ...nextPath, count: item.count });
        }
    });

    return rows;
}

function renderResultGrid(data) {
    initGrid();

    const breakdown = [...activeBreakdown];
    const rows = flattenResult(data, breakdown);
    const columnDefs = breakdown.map(column => ({
        field: column,
        headerName: column,
        sortable: true,
        resizable: true,
        filter: true,
        flex: 1,
        minWidth: 140
    }));
    columnDefs.push({
        field: 'count',
        headerName: 'Count',
        sortable: true,
        resizable: true,
        filter: true,
        flex: 1,
        minWidth: 120,
        sort: 'desc'
    });

    if (!gridApi) {
        setTimeout(() => renderResultGrid(data), 0);
        return;
    }

    gridApi.setColumnDefs(columnDefs);
    gridApi.setRowData(rows);
}

function renderResult(data) {
    latestResult = data;
    renderResultGrid(data);
    document.getElementById('result-json').innerText = JSON.stringify(data, null, 2);
    setResultMode(resultMode);
}

function getAISummary() {
    const rows = flattenResult(latestResult, [...activeBreakdown]);
    const topValues = rows
        .map(row => {
            const keys = Object.keys(row).filter(key => key !== 'count');
            const valueKey = keys[keys.length - 1];
            return {
                value: valueKey ? row[valueKey] : null,
                count: row.count
            };
        })
        .filter(item => item.value !== null && item.value !== undefined && item.value !== '')
        .slice(0, 10);

    return {
        breakdown: [...activeBreakdown],
        top_values: topValues
    };
}

function openCharts() {
    localStorage.setItem("chartData", JSON.stringify(latestResult));
    localStorage.setItem("chartBreakdown", JSON.stringify(activeBreakdown));
    window.location.href = "/charts";
}

function renderAIMessages() {
    const list = document.getElementById('ai-messages');
    const status = document.getElementById('ai-status');
    list.innerHTML = '';
    status.innerText = '';

    if (!aiMessages.length) {
        list.innerText = 'No response yet';
        return;
    }

    aiMessages.forEach(message => {
        const item = document.createElement('div');
        item.className = 'chat-message';
        item.innerText = `${message.role}: ${message.text}`;
        list.appendChild(item);
    });
}

function normalizeValue(value) {
    return String(value ?? '')
        .trim()
        .toLowerCase()
        .replaceAll('ё', 'е')
        .replace(/\\s{2,}/g, ' ');
}

async function loadUniqueValues(column) {
    const valuesSelect = document.getElementById('values');
    valuesSelect.innerHTML = '';

    if (!column) {
        return;
    }

    const res = await fetch('/unique_values', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ column: column })
    });
    const values = await res.json();

    values.forEach(value => {
        const opt = document.createElement('option');
        opt.value = value;
        opt.text = value;
        valuesSelect.appendChild(opt);
    });
}

async function normalizeSelectedColumn() {
    const column = document.getElementById('columns').value;
    if (!column) {
        return;
    }

    const button = document.getElementById('normalize-button');
    const status = document.getElementById('normalize-status');
    button.disabled = true;
    status.innerText = 'processing...';

    try {
        const res = await fetch('/normalize_column', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ column: column })
        });

        if (res.ok) {
            await loadUniqueValues(column);
            await loadDatasetPage(datasetPage);
            await group();
            status.innerText = 'done';
        } else {
            status.innerText = 'failed';
        }
    } finally {
        button.disabled = false;
    }
}

async function addFilter(column, value) {
    activeFilters[column] = value;
    renderFilters();
    await group();
}

async function removeFilter(column) {
    delete activeFilters[column];
    renderFilters();
    await group();
}

async function addSelectedFilter() {
    const column = document.getElementById('columns').value;
    const value = document.getElementById('values').value;
    if (column && value) {
        await addFilter(column, value);
    }
}

async function clearFilters() {
    activeFilters = {};
    renderFilters();
    await group();
}

async function addBreakdown() {
    const column = document.getElementById('breakdown-column').value;
    if (!column) {
        return;
    }
    if (!activeBreakdown.includes(column)) {
        activeBreakdown.push(column);
        renderBreakdown();
    }
    await group();
}

async function moveBreakdownUp(index) {
    if (index <= 0) {
        return;
    }
    [activeBreakdown[index - 1], activeBreakdown[index]] = [activeBreakdown[index], activeBreakdown[index - 1]];
    renderBreakdown();
    await group();
}

async function moveBreakdownDown(index) {
    if (index >= activeBreakdown.length - 1) {
        return;
    }
    [activeBreakdown[index], activeBreakdown[index + 1]] = [activeBreakdown[index + 1], activeBreakdown[index]];
    renderBreakdown();
    await group();
}

async function removeBreakdown(index) {
    activeBreakdown.splice(index, 1);
    renderBreakdown();
    await group();
}

async function clearBreakdown() {
    activeBreakdown = [];
    renderBreakdown();
    await group();
}

async function sendAIMessage() {
    const input = document.getElementById('ai-input');
    const message = input.value.trim();
    if (!message) {
        return;
    }

    const button = document.getElementById('ai-button');
    const status = document.getElementById('ai-status');
    button.disabled = true;
    status.innerText = 'processing...';
    aiMessages.push({ role: 'User', text: message });
    renderAIMessages();
    input.value = '';

    try {
        const res = await fetch('/ai/query', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                message: message,
                summary: getAISummary()
            })
        });

        if (!res.ok) {
            const errorText = await res.text();
            aiMessages.push({ role: 'AI', text: errorText || 'No response yet' });
            renderAIMessages();
            return;
        }

        const data = await res.json();
        if (!data.text) {
            aiMessages.push({ role: 'AI', text: 'No response yet' });
        } else {
            aiMessages.push({ role: 'AI', text: data.text });
        }
        renderAIMessages();
    } finally {
        button.disabled = false;
        status.innerText = '';
    }
}

async function group() {
    const currentFilters = { ...activeFilters };
    const currentBreakdown = [...activeBreakdown];
    const res = await fetch('/analyze', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            breakdown: currentBreakdown,
            filters: currentFilters
        })
    });
    const data = await res.json();
    renderResult(data);
}

initGrid();
initDatasetGrid();
renderFilters();
renderBreakdown();
renderAIMessages();
updateDatasetPagination();
setResultMode('table');
initializeLoadedDataset();
</script>
</body>
</html>
""",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/charts", response_class=HTMLResponse)
def charts_page():
    return HTMLResponse(
        """<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>Charts</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
    * { box-sizing: border-box; }
    body {
        margin: 0;
        padding: 24px;
        font-family: Arial, sans-serif;
        background: #f9fafb;
        color: #111827;
    }
    .wrap {
        max-width: 1200px;
        margin: 0 auto;
    }
    .toolbar {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        align-items: end;
        margin-bottom: 16px;
    }
    .field {
        min-width: 180px;
    }
    label {
        display: block;
        font-size: 13px;
        margin-bottom: 6px;
        color: #4b5563;
    }
    select, button {
        padding: 8px 10px;
        border: 1px solid #e5e7eb;
        border-radius: 4px;
        background: #fff;
        font: inherit;
    }
    button {
        background: #f3f4f6;
        cursor: pointer;
    }
    .chart-box {
        background: #fff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px;
    }
    canvas {
        width: 100%;
        max-width: 100%;
        height: 480px;
    }
    </style>
</head>
<body>
    <div class="wrap">
        <div style="margin-bottom:16px;">
            <a href="/app">Back to app</a>
        </div>
        <div class="toolbar">
            <div class="field">
                <label for="chart-type">Chart type</label>
                <select id="chart-type">
                    <option value="bar">bar</option>
                    <option value="line">line</option>
                    <option value="pie">pie</option>
                </select>
            </div>
            <div class="field">
                <label for="chart-x">X-axis</label>
                <select id="chart-x"></select>
            </div>
            <div class="field">
                <label for="chart-y">Y-axis</label>
                <select id="chart-y"></select>
            </div>
        </div>
        <div class="chart-box">
            <canvas id="chart-canvas"></canvas>
        </div>
    </div>
    <script>
    let chartInstance = null;

    function flattenResult(items, breakdown, path = {}) {
        if (!Array.isArray(items)) {
            if (items && typeof items.count === 'number') {
                return [{ ...path, count: items.count }];
            }
            return [];
        }

        const levelIndex = Object.keys(path).length;
        const currentKey = breakdown[levelIndex] || `Level ${levelIndex + 1}`;
        let rows = [];

        items.forEach(item => {
            const nextPath = { ...path, [currentKey]: item.group };
            if (Array.isArray(item.subgroups) && item.subgroups.length) {
                rows.push(...flattenResult(item.subgroups, breakdown, nextPath));
            } else {
                rows.push({ ...nextPath, count: item.count });
            }
        });

        return rows;
    }

    function getRows() {
        const chartData = JSON.parse(localStorage.getItem('chartData') || '[]');
        const chartBreakdown = JSON.parse(localStorage.getItem('chartBreakdown') || '[]');
        return flattenResult(chartData, chartBreakdown);
    }

    function initFields() {
        const rows = getRows();
        if (!rows.length) {
            return;
        }
        const keys = Object.keys(rows[0]).filter(k => k !== 'count');
        const xSelect = document.getElementById('chart-x');
        const ySelect = document.getElementById('chart-y');
        xSelect.innerHTML = '';
        ySelect.innerHTML = '';

        keys.forEach(column => {
            const xOpt = document.createElement('option');
            xOpt.value = column;
            xOpt.text = column;
            xSelect.appendChild(xOpt);
        });

        [...keys, 'count'].forEach(column => {
            const yOpt = document.createElement('option');
            yOpt.value = column;
            yOpt.text = column;
            ySelect.appendChild(yOpt);
        });

        xSelect.value = keys[0] || '';
        ySelect.value = 'count';
    }

    function buildChart() {
        const rows = getRows();
        if (!rows.length) {
            return;
        }
        const type = document.getElementById('chart-type').value;
        const xKey = document.getElementById('chart-x').value;
        const yKey = document.getElementById('chart-y').value;
        const labels = rows.map(row => row[xKey]);
        const data = rows.map(row => Number(row[yKey]) || 0);
        const ctx = document.getElementById('chart-canvas');

        if (chartInstance) {
            chartInstance.destroy();
        }

        chartInstance = new Chart(ctx, {
            type: type,
            data: {
                labels: labels,
                datasets: [{
                    label: yKey,
                    data: data,
                    backgroundColor: 'rgba(59, 130, 246, 0.5)',
                    borderColor: 'rgba(59, 130, 246, 1)',
                    borderWidth: 1,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
            }
        });
    }

    initFields();
    document.getElementById('chart-type').addEventListener('change', buildChart);
    document.getElementById('chart-x').addEventListener('change', buildChart);
    document.getElementById('chart-y').addEventListener('change', buildChart);
    buildChart();
    </script>
</body>
</html>"""
    )


@app.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...), tags: str = Form(default="")):
    global current_df, current_file_name, current_dataset_id

    if not file.filename or not file.filename.lower().endswith((".csv", ".xlsx")):
        raise HTTPException(status_code=400, detail="Only CSV and XLSX files are supported")

    data_dir = ensure_data_dir()
    content = await file.read()
    dataset_id = uuid.uuid4().hex
    saved_file_path = data_dir / f"{dataset_id}_{file.filename}"
    await run_in_threadpool(save_file, saved_file_path, content)
    current_df = read_table_from_path(saved_file_path)
    current_file_name = file.filename
    current_dataset_id = dataset_id

    datasets = load_datasets()
    datasets.append(
        {
            "id": dataset_id,
            "name": file.filename,
            "file_path": str(saved_file_path),
            "created_at": datetime.utcnow().isoformat(),
            "tags": parse_tags(tags),
        }
    )
    save_datasets(datasets)
    return RedirectResponse(url=f"/app?dataset_id={dataset_id}", status_code=303)


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global current_df, upload_status, current_file_name, current_dataset_id

    if not file.filename or not file.filename.lower().endswith((".csv", ".xlsx")):
        raise HTTPException(status_code=400, detail="Only CSV and XLSX files are supported")

    try:
        print("start upload")
        upload_status = {"status": "uploading", "progress": 10}
        data_dir = ensure_data_dir()
        content = await file.read()
        print(f"filename: {file.filename}")
        print(f"size: {len(content)}")
        file_path = data_dir / file.filename
        await run_in_threadpool(save_file, file_path, content)
        print("file saved")

        upload_status = {"status": "processing", "progress": 60}
        print("reading file")
        file.file.seek(0)
        if hasattr(file.file, "write"):
            if hasattr(file.file, "truncate"):
                file.file.truncate(0)
                file.file.seek(0)
            file.file.write(content)
            file.file.seek(0)
        else:
            file.file.seek(0)
        current_df = await run_in_threadpool(read_table, file)
        current_file_name = file.filename
        current_dataset_id = None
        preview = current_df.head(50).astype(object).where(pd.notna(current_df.head(50)), None)
        upload_status = {"status": "done", "progress": 100}
        print("done")

        return preview.to_dict(orient="records")
    except Exception as error:
        print(f"Upload error: {error}")
        upload_status = {"status": "error", "progress": 0, "error": str(error)}
        return JSONResponse(status_code=400, content={"error": str(error)})


@app.get("/status")
async def get_status():
    return upload_status


@app.get("/current_dataset")
async def current_dataset():
    if current_df is None:
        return {"id": None, "name": None, "row_count": 0, "column_count": 0}

    return {
        "id": current_dataset_id,
        "name": current_file_name,
        "row_count": int(len(current_df)),
        "column_count": int(len(current_df.columns)),
    }


@app.get("/columns")
async def get_columns():
    if current_df is None:
        raise HTTPException(status_code=400, detail="No data")

    return {
        "columns": [
            {"index": index, "name": name}
            for index, name in enumerate(current_df.columns.tolist())
        ]
    }


@app.get("/dataset_rows")
async def dataset_rows(page: int = 1, page_size: int = 100):
    if current_df is None:
        return {"rows": [], "total_rows": 0, "columns": []}

    safe_page = max(page, 1)
    safe_page_size = max(min(page_size, 1000), 1)
    return {
        "rows": dataframe_page_to_records(current_df, safe_page, safe_page_size),
        "total_rows": int(len(current_df)),
        "columns": current_df.columns.tolist(),
    }


@app.post("/count")
async def count_rows(request: CountRequest):
    if current_df is None:
        raise HTTPException(status_code=400, detail="No file uploaded")

    return {"count": len(current_df)}


@app.post("/unique_values")
async def unique_values(request: UniqueValuesRequest):
    if current_df is None:
        raise HTTPException(status_code=400, detail="No file uploaded")

    if request.column.isdigit():
        col_data = current_df.iloc[:, int(request.column)]
    else:
        col_data = current_df[request.column]

    values = (
        col_data.dropna()
        .astype(str)
        .loc[lambda s: s.str.strip() != ""]
        .drop_duplicates()
        .head(1000)
        .tolist()
    )
    return values


@app.post("/dataset/{dataset_id}/tags")
async def update_dataset_tags(dataset_id: str, tags: str = Form(default="")):
    datasets = load_datasets()
    updated = False
    for dataset in datasets:
        if dataset["id"] == dataset_id:
            dataset["tags"] = parse_tags(tags)
            updated = True
            break

    if not updated:
        raise HTTPException(status_code=404, detail="Dataset not found")

    save_datasets(datasets)
    return {"status": "ok"}


@app.post("/group")
async def group_rows(request: GroupRequest):
    if current_df is None:
        raise HTTPException(status_code=400, detail="No file uploaded")

    df = apply_filters(current_df.copy(), request.filters)
    breakdown = request.breakdown or []

    if not breakdown and request.group_by:
        if request.group_by == "value" and request.column:
            breakdown = [request.column]
        else:
            raise HTTPException(status_code=400, detail="Invalid group_by value")

    if not breakdown:
        return {"count": int(len(df))}

    result = build_group_tree(df, breakdown)
    if request.group_by == "value" and isinstance(result, list):
        return sort_groups_by_count(result)
    return result


@app.post("/analyze")
async def analyze_rows(request: AnalyzeRequest):
    if current_df is None:
        raise HTTPException(status_code=400, detail="No file uploaded")

    df = apply_filters(current_df.copy(), request.filters)
    breakdown = request.breakdown or []

    if not breakdown:
        return {"count": int(len(df))}

    return sort_group_tree(build_group_tree(df, breakdown))


@app.post("/normalize_column")
async def normalize_column(request: NormalizeColumnRequest):
    global current_df

    if current_df is None:
        raise HTTPException(status_code=400, detail="No file uploaded")

    column_key = get_column_key(current_df, request.column)
    current_df[column_key] = (
        current_df[column_key]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("ё", "е", regex=False)
        .str.replace(r"\s{2,}", " ", regex=True)
    )

    return {"status": "ok"}


@app.post("/ai/suggest")
async def ai_suggest(request: AISuggestRequest):
    return {"suggestions": []}


@app.post("/ai/query")
async def ai_query(request: AIQueryRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not set")

    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = request.message
    if request.summary:
        prompt = (
            f"{request.message}\n\n"
            f"Context summary:\n{json.dumps(request.summary, ensure_ascii=False)}"
        )
    response = model.generate_content(prompt)

    return {
        "type": "analysis",
        "text": response.text or "",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))

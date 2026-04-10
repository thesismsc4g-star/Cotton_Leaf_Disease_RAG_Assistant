import json
from pathlib import Path
from typing import Iterable, List

from rag_pipeline import build_vectorstore

DEFAULT_URLS = [
    "https://www.cottoninc.com/cotton-production/ag-research/plant-pathology/management-bacterial-blight-cotton/",
    "https://content.ces.ncsu.edu/bacterial-blight-angular-leaf-spot-of-cotton",
    "https://news.utcrops.com/2021/08/cotton-diseases-and-management-options/",
    "https://www.uaex.uada.edu/farm-ranch/pest-management/plant-disease/cotton.aspx",
    "https://ipm.ucanr.edu/agriculture/cotton/fusarium-wilt/",
    "https://cropprotectionnetwork.org/encyclopedia/fusarium-wilt-of-cotton",
    "https://ipm.ucanr.edu/agriculture/cotton/verticillium-wilt/",
    "https://cropprotectionnetwork.org/encyclopedia/verticillium-wilt-of-cotton",
    "https://cropprotectionnetwork.org/encyclopedia/alternaria-leaf-spot-of-cotton",
    "https://guide.utcrops.com/cotton/cotton-foliar-diseases/alternaria-leaf-spot/",
    "https://acis.cals.arizona.edu/docs/default-source/agricultural-ipm-documents/cotton/az1854-2020.pdf?sfvrsn=ccd19221_0",
]


def _normalize_urls(urls: Iterable[str]) -> List[str]:
    normalized = []
    for url in urls:
        if isinstance(url, str):
            cleaned = url.strip()
            if cleaned:
                normalized.append(cleaned)
    return normalized


def ensure_sources_json() -> None:
    data_dir = Path(__file__).resolve().parent / "data"
    sources_path = data_dir / "sources.json"
    data_dir.mkdir(parents=True, exist_ok=True)

    data = {}
    if sources_path.exists():
        try:
            data = json.loads(sources_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}

    existing = _normalize_urls(data.get("urls", []))
    merged = []
    seen = set()
    for url in existing + DEFAULT_URLS:
        if url not in seen:
            merged.append(url)
            seen.add(url)

    if data.get("urls") != merged:
        sources_path.write_text(
            json.dumps({"urls": merged}, indent=2), encoding="utf-8"
        )


def main() -> None:
    ensure_sources_json()
    build_vectorstore()
    print("Vector store built in ./vectorstore")


if __name__ == "__main__":
    main()

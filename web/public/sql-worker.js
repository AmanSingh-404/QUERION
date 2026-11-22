/* global importScripts, initSqlJs */
importScripts("/sql-wasm.js");

let db = null;
let ready = null;

function init() {
  if (!ready) ready = initSqlJs({ locateFile: (file) => "/" + file });
  return ready;
}

self.onmessage = async (event) => {
  const { id, type, payload } = event.data;
  try {
    const SQL = await init();
    if (type === "load") {
      if (db) db.close();
      db = new SQL.Database();
      db.run(payload.create);
      const insert = db.prepare(payload.insert);
      db.run("BEGIN");
      for (const row of payload.rows) insert.run(row);
      db.run("COMMIT");
      insert.free();
      self.postMessage({ id, ok: true });
    } else if (type === "run") {
      const results = db.exec(payload.sql);
      const last = results[results.length - 1];
      const result = last
        ? { columns: last.columns, rows: last.values }
        : { columns: [], rows: [] };
      self.postMessage({ id, ok: true, result });
    }
  } catch (e) {
    self.postMessage({ id, ok: false, error: String(e && e.message ? e.message : e) });
  }
};
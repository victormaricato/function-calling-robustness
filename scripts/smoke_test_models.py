"""Single-call smoke test on every model slug in models.SOTA + EFFORT_SWEEP.

Confirms slug resolves, tool-call API accepts payload, reasoning param doesn't
404. Faster to fail here ($0.01) than 30 minutes into a 28K-cell run.
"""

from __future__ import annotations

import asyncio
import sys
import time

from stale_tools.harness.models import EFFORT_SWEEP, NEW_SOTA, SOTA, ModelSpec
from stale_tools.harness.runner import _client

TRIVIAL_TOOL = {
    "type": "function",
    "function": {
        "name": "echo",
        "description": "Echo a short string back to the caller.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
}


async def probe(client, m: ModelSpec) -> dict:
    extra_body: dict = {"provider": {"allow_fallbacks": True}}
    if m.extra_body:
        for k, v in m.extra_body.items():
            if k in extra_body and isinstance(extra_body[k], dict) and isinstance(v, dict):
                extra_body[k].update(v)
            else:
                extra_body[k] = v
    t0 = time.time()
    try:
        resp = await client.chat.completions.create(
            model=m.slug,
            messages=[
                {"role": "system", "content": "Use the echo tool when asked."},
                {"role": "user", "content": "Echo the word 'ok'."},
            ],
            tools=[TRIVIAL_TOOL],
            tool_choice="auto",
            temperature=0,
            max_tokens=128,
            extra_body=extra_body,
        )
        ok = bool(getattr(resp, "choices", None))
        if not ok:
            err = getattr(resp, "error", None) or {}
            return {
                "slug": m.slug,
                "nick": m.nickname,
                "ok": False,
                "err": err.get("message", str(resp)),
                "dt": time.time() - t0,
            }
        msg = resp.choices[0].message
        called = [tc.function.name for tc in (msg.tool_calls or [])]
        return {
            "slug": m.slug,
            "nick": m.nickname,
            "ok": True,
            "called": called,
            "provider": resp.model,
            "dt": time.time() - t0,
        }
    except Exception as e:
        return {
            "slug": m.slug,
            "nick": m.nickname,
            "ok": False,
            "err": f"{type(e).__name__}: {e}",
            "dt": time.time() - t0,
        }


async def main():
    client = _client()
    if "--only-new" in sys.argv:
        targets: list[ModelSpec] = list(NEW_SOTA)
        print(f"probing {len(targets)} unique slugs...")
        results = await asyncio.gather(*(probe(client, m) for m in targets))
        fail = 0
        for r in results:
            if r["ok"]:
                print(
                    f"  OK  {r['nick']:24s} {r['slug']:50s} called={r.get('called')} ({r['dt']:.1f}s)"
                )
            else:
                fail += 1
                print(f"  FAIL {r['nick']:24s} {r['slug']:50s} {r['err']!r}")
        print(f"\n{len(results) - fail}/{len(results)} OK, {fail} failed")
        return fail
    # Dedupe by slug for SOTA + 4 representative effort variants per family
    targets: list[ModelSpec] = list(SOTA) + list(NEW_SOTA)
    seen_slugs = {m.slug for m in targets}
    for m in EFFORT_SWEEP:
        # one representative per (slug, effort=med) only — covers the unique payload
        if m.effort == "med" and m.slug not in seen_slugs:
            targets.append(m)
            seen_slugs.add(m.slug)
    # Always probe the OpenAI 'minimal' variant since that's a new code path.
    for m in EFFORT_SWEEP:
        if m.effort == "off" and "openai" in m.slug:
            targets.append(m)
            break

    print(f"probing {len(targets)} unique slugs...")
    results = await asyncio.gather(*(probe(client, m) for m in targets))
    fail = 0
    for r in results:
        if r["ok"]:
            print(
                f"  OK  {r['nick']:24s} {r['slug']:50s} called={r.get('called')} ({r['dt']:.1f}s)"
            )
        else:
            fail += 1
            print(f"  FAIL {r['nick']:24s} {r['slug']:50s} {r['err']!r}")
    print(f"\n{len(results) - fail}/{len(results)} OK, {fail} failed")
    return fail


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

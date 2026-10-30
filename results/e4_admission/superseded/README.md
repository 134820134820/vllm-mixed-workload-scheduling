First runs of `gw_fifo` and `gw_sjf` at 7.5 req/s. Each had one HTTP 500,
returned within a few milliseconds and before the gateway sent any response
header; the server-side counters show the two requests never reached vLLM.
The gateway log of that run showed `aiohttp.ServerDisconnectedError` from the
upstream session, consistent with reusing an idle keep-alive connection just
as vLLM closed it. That log was overwritten when the cases were re-run (the
harness now appends instead). The gateway now disables upstream keep-alive and answers
502 on upstream errors. These two cases were re-run and are kept here for the
record; they are not used in summaries.

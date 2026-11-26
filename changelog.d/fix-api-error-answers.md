- Fixed: an address under the API that does not exist now answers "Not Found"
  instead of returning the app's page or "Method Not Allowed".
- Fixed: after a server error, the next request on the same connection no
  longer fails; the error answer now tells the client the connection closes.
- Fixed: a settings, person, picture or guest-score change whose body is not
  valid JSON now answers with a clear error instead of a server error, and a
  picture change no longer reports success after ignoring such a body.

from http.server import HTTPServer, BaseHTTPRequestHandler

from io import BytesIO

import re


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'Hello, world!')

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        self.send_response(200)
        self.end_headers()
        response = BytesIO()
        response.write(b'This is POST request. ')
        response.write(b'Received: ')
        response.write(body)
        self.wfile.write(response.getvalue())

        data = re.split(',',body.decode("utf-8"))
        enc1 = data[4]
        enc2 = re.split('}',data[5])[0].strip()
        print(enc1+"    "+enc2)

    def log_message(self, format, *args):
        return


httpd = HTTPServer(('0.0.0.0', 8000), SimpleHTTPRequestHandler)
httpd.serve_forever()
#include <stdio.h>
#include <string.h>
#include <memory.h>
#include <stdlib.h>
using namespace std; 

const char* quote_identifier(const char* ident)
{
  /*
   * Can avoid quoting if ident starts with a lowercase letter or underscore
   * and contains only lowercase letters, digits, and underscores, *and* is
   * not any SQL keyword.  Otherwise, supply quotes.
   */
  int         nquotes = 0;
  bool        safe;
  const char *ptr;
  char       *result;
  char       *optr;
  /*
   * would like to use <ctype.h> macros here, but they might yield unwanted
   * locale-specific results...
   */
  safe = ((ident[0] >= 'a' && ident[0] <= 'z') || ident[0] == '_');
  for (ptr = ident; *ptr; ptr++)
  {
      char        ch = *ptr;
      if ((ch >= 'a' && ch <= 'z') ||
          (ch >= '0' && ch <= '9') ||
          (ch == '_'))
      {
          /* okay */
      }
      else
      {
          safe = false;
          if (ch == '"')
              nquotes++;
      }
  }
  /*if (quote_all_identifiers)
      safe = false;
  if (safe)
  {
      const ScanKeyword *keyword = ScanKeywordLookup(ident,
                                                     ScanKeywords,
                                                     NumScanKeywords);
      if (keyword != NULL && keyword->category != UNRESERVED_KEYWORD)
          safe = false;
  }
  if (safe)
      return ident;           /* no change needed */
  result = (char *) malloc(strlen(ident) + nquotes + 2 + 1);
  optr = result;
  *optr++ = '"';
  for (ptr = ident; *ptr; ptr++)
  {
      char        ch = *ptr;
      if (ch == '"')
          *optr++ = '"';
      *optr++ = ch;
  }
  *optr++ = '"';
  *optr = '\0';
  return result;

}


int main(int argc, char const *argv[])
{
  char*  str = "\'foo bar\'";
  const char* qstr = quote_identifier(str);
  printf("%s\n", qstr);
  return 0;
}
  
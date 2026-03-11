/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "zep/syntax_python.h"

#include "zep/editor.h"
#include "zep/theme.h"

#include "zep/mcommon/string/stringutils.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>

namespace Zep
{

namespace
{

bool is_identifier_start(uint8_t ch)
{
    return std::isalpha(static_cast<unsigned char>(ch)) != 0 || ch == '_';
}

bool is_identifier_continue(uint8_t ch)
{
    return std::isalnum(static_cast<unsigned char>(ch)) != 0 || ch == '_';
}

bool is_python_string_prefix(std::string_view prefix)
{
    const std::string lowered = string_tolower(std::string(prefix));
    static constexpr std::string_view valid_prefixes[] = {
        "r", "u", "b", "f", "rb", "br", "rf", "fr",
    };

    return std::find(std::begin(valid_prefixes), std::end(valid_prefixes), lowered) !=
           std::end(valid_prefixes);
}

} // namespace

ZepSyntax_Python::ZepSyntax_Python(ZepBuffer& buffer,
    const std::unordered_set<std::string>& keywords,
    const std::unordered_set<std::string>& identifiers,
    uint32_t flags)
    : ZepSyntax(buffer, keywords, identifiers, flags)
{
}

void ZepSyntax_Python::UpdateSyntax()
{
    auto& buffer = m_buffer.GetWorkingBuffer();
    auto itrCurrent = buffer.begin();
    const auto itrEnd = buffer.end();

    assert(std::distance(itrCurrent, itrEnd) <= int(m_syntax.size()));
    assert(m_syntax.size() == buffer.size());

    auto mark = [&](GapBuffer<uint8_t>::const_iterator itrA,
                    GapBuffer<uint8_t>::const_iterator itrB,
                    ThemeColor type,
                    ThemeColor background) {
        std::fill(m_syntax.begin() + (itrA - buffer.begin()),
                  m_syntax.begin() + (itrB - buffer.begin()),
                  SyntaxData{type, background});
    };

    auto mark_single = [&](GapBuffer<uint8_t>::const_iterator itr,
                           ThemeColor type,
                           ThemeColor background) {
        (m_syntax.begin() + (itr - buffer.begin()))->foreground = type;
        (m_syntax.begin() + (itr - buffer.begin()))->background = background;
    };

    auto scan_string = [&](auto start, auto quote_begin) {
        auto itr = quote_begin;
        const uint8_t quote = *quote_begin;
        bool triple = false;
        if ((itrEnd - quote_begin) >= 3 && *(quote_begin + 1) == quote &&
            *(quote_begin + 2) == quote) {
            triple = true;
            itr += 3;
        } else {
            ++itr;
        }

        while (itr != itrEnd) {
            if (!triple && *itr == '\\') {
                if ((itrEnd - itr) > 1) {
                    itr += 2;
                } else {
                    ++itr;
                }
                continue;
            }

            if (*itr == quote) {
                if (triple) {
                    if ((itrEnd - itr) >= 3 && *(itr + 1) == quote && *(itr + 2) == quote) {
                        itr += 3;
                        break;
                    }
                } else {
                    ++itr;
                    break;
                }
            }

            if (!triple && *itr == '\n') {
                break;
            }

            ++itr;
        }

        mark(start, itr, ThemeColor::String, ThemeColor::None);
        return itr;
    };

    std::fill(m_syntax.begin(), m_syntax.end(), SyntaxData{});

    while (itrCurrent != itrEnd) {
        if (m_stop) {
            return;
        }

        m_processedChar = long(itrCurrent - buffer.begin());
        const uint8_t ch = *itrCurrent;

        if (ch == ' ' || ch == '\t') {
            mark_single(itrCurrent, ThemeColor::Whitespace, ThemeColor::None);
            ++itrCurrent;
            continue;
        }

        if (ch == '\n' || ch == '\r' || ch == 0) {
            ++itrCurrent;
            continue;
        }

        if (ch == '#') {
            const auto comment_start = itrCurrent;
            while (itrCurrent != itrEnd && *itrCurrent != '\n' && *itrCurrent != 0) {
                ++itrCurrent;
            }
            mark(comment_start, itrCurrent, ThemeColor::Comment, ThemeColor::None);
            continue;
        }

        if (ch == '\'' || ch == '"') {
            itrCurrent = scan_string(itrCurrent, itrCurrent);
            continue;
        }

        if (is_identifier_start(ch)) {
            const auto token_start = itrCurrent;
            auto token_end = itrCurrent + 1;
            while (token_end != itrEnd && is_identifier_continue(*token_end)) {
                ++token_end;
            }

            const std::string prefix(token_start, token_end);
            if (token_end != itrEnd && (*token_end == '\'' || *token_end == '"') &&
                is_python_string_prefix(prefix)) {
                itrCurrent = scan_string(token_start, token_end);
                continue;
            }

            std::string token = prefix;
            if (m_flags & ZepSyntaxFlags::CaseInsensitive) {
                token = string_tolower(token);
            }

            ThemeColor color = ThemeColor::Normal;
            if (m_keywords.find(token) != m_keywords.end()) {
                color = ThemeColor::Keyword;
            } else if (m_identifiers.find(token) != m_identifiers.end() || token == "self" ||
                       token == "cls") {
                color = ThemeColor::Identifier;
            }

            mark(token_start, token_end, color, ThemeColor::None);
            itrCurrent = token_end;
            continue;
        }

        if (std::isdigit(static_cast<unsigned char>(ch)) != 0 ||
            (ch == '.' && (itrCurrent + 1) != itrEnd &&
             std::isdigit(static_cast<unsigned char>(*(itrCurrent + 1))) != 0)) {
            const auto number_start = itrCurrent;
            ++itrCurrent;

            while (itrCurrent != itrEnd) {
                const uint8_t current = *itrCurrent;
                if (std::isalnum(static_cast<unsigned char>(current)) != 0 || current == '_' ||
                    current == '.') {
                    ++itrCurrent;
                    continue;
                }

                if ((current == '+' || current == '-') && itrCurrent != number_start) {
                    const uint8_t previous = *(itrCurrent - 1);
                    if (previous == 'e' || previous == 'E') {
                        ++itrCurrent;
                        continue;
                    }
                }

                break;
            }

            mark(number_start, itrCurrent, ThemeColor::Number, ThemeColor::None);
            continue;
        }

        if (ch == '@') {
            const auto decorator_start = itrCurrent;
            ++itrCurrent;
            while (itrCurrent != itrEnd && is_identifier_continue(*itrCurrent)) {
                ++itrCurrent;
            }
            mark(decorator_start, itrCurrent, ThemeColor::Identifier, ThemeColor::None);
            continue;
        }

        if (ch == '(' || ch == ')' || ch == '[' || ch == ']' || ch == '{' || ch == '}') {
            mark_single(itrCurrent, ThemeColor::Parenthesis, ThemeColor::None);
        }

        ++itrCurrent;
    }

    m_targetChar = 0;
    m_processedChar = buffer.size() == 0 ? 0 : long(buffer.size() - 1);
}

} // namespace Zep

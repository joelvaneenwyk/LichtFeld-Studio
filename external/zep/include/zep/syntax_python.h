/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "syntax.h"

namespace Zep
{

class ZepSyntax_Python : public ZepSyntax
{
public:
    ZepSyntax_Python(ZepBuffer& buffer,
        const std::unordered_set<std::string>& keywords = std::unordered_set<std::string>{},
        const std::unordered_set<std::string>& identifiers = std::unordered_set<std::string>{},
        uint32_t flags = 0);

    void UpdateSyntax() override;
};

} // namespace Zep

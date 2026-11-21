
local lastMessage = "Waiting..."

local function run(event)
    local message = serialRead()

    if message and #message > 0 then
        lastMessage = message
        serialWrite("ACK:" .. message .. "\n")
    end

    lcd.clear()
    lcd.drawText(2, 2, "AutoBots USB Test", 0)
    lcd.drawText(2, 20, string.sub(lastMessage, 1, 20), 0)

    return 0
end

return { run = run }
